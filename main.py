"""
Filename: main.py
Author: Boyang
Date: 2025-11-15
Lines: 351
Description: Main training and evaluation script for RepViT model.
"""

import os
import json
import time
import torch
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from data.samplers import RASampler
from data.datasets import build_dataset
from data.threeaugment import new_data_aug_generator
from engine import train_one_epoch, evaluate

import model
import utils
import wandb


def create_argument_parser():
    """Create and configure argument parser for training/evaluation."""
    arg_parser = argparse.ArgumentParser('RepViT training and evaluation script', add_help=False)
    arg_parser.add_argument('--batch-size', default=256, type=int)
    arg_parser.add_argument('--epochs', default=300, type=int)

    arg_parser.add_argument('--model', default='repvit_m1_1', type=str, metavar='MODEL',
                            help='Name of model to train')
    arg_parser.add_argument('--input-size', default=224, type=int, help='images input size')

    arg_parser.add_argument('--model-ema', action='store_true')
    arg_parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    arg_parser.set_defaults(model_ema=True)
    arg_parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    arg_parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    arg_parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                            help='Optimizer (default: "adamw"')
    arg_parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                            help='Optimizer Epsilon (default: 1e-8)')
    arg_parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                            help='Optimizer Betas (default: None, use opt default)')
    arg_parser.add_argument('--clip-grad', type=float, default=0.02, metavar='NORM',
                            help='Clip gradient norm (default: None, no clipping)')
    arg_parser.add_argument('--clip-mode', type=str, default='agc',
                            help='Gradient clipping mode. One of ("norm", "value", "agc")')
    arg_parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='SGD momentum (default: 0.9)')
    arg_parser.add_argument('--weight-decay', type=float, default=0.025, help='weight decay (default: 0.025)')

    arg_parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                            help='LR scheduler (default: "cosine"')
    arg_parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    arg_parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                            help='learning rate noise on/off epoch percentages')
    arg_parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                            help='learning rate noise limit percent (default: 0.67)')
    arg_parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                            help='learning rate noise std-dev (default: 1.0)')
    arg_parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                            help='warmup learning rate (default: 1e-6)')
    arg_parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    arg_parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                            help='epoch interval to decay LR')
    arg_parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                            help='epochs to warmup LR, if scheduler supports')
    arg_parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                            help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    arg_parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                            help='patience epochs for Plateau LR scheduler (default: 10')
    arg_parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                            help='LR decay rate (default: 0.1)')

    arg_parser.add_argument('--ThreeAugment', action='store_true')
    arg_parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                            help='Color jitter factor (default: 0.4)')
    arg_parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + \
                                 "(default: rand-m9-mstd0.5-inc1)')
    arg_parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    arg_parser.add_argument('--train-interpolation', type=str, default='bicubic',
                            help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    arg_parser.add_argument('--repeated-aug', action='store_true')
    arg_parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    arg_parser.set_defaults(repeated_aug=True)

    arg_parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
    arg_parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    arg_parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')
    arg_parser.add_argument('--resplit', action='store_true', default=False,
                            help='Do not random erase first (clean) augmentation split')

    arg_parser.add_argument('--mixup', type=float, default=0.8,
                            help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    arg_parser.add_argument('--cutmix', type=float, default=1.0,
                            help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    arg_parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                            help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    arg_parser.add_argument('--mixup-prob', type=float, default=1.0,
                            help='Probability of performing mixup or cutmix when either/both is enabled')
    arg_parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                            help='Probability of switching to cutmix when both mixup and cutmix enabled')
    arg_parser.add_argument('--mixup-mode', type=str, default='batch',
                            help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    arg_parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    arg_parser.add_argument('--set_bn_eval', action='store_true', default=False,
                            help='set BN layers to eval mode during finetuning.')

    arg_parser.add_argument('--data-path', default='/root/FastBaseline/data/imagenet', type=str,
                            help='dataset path')
    arg_parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                            type=str, help='Image Net dataset path')
    arg_parser.add_argument('--inat-category', default='name',
                            choices=['kingdom', 'phylum', 'class', 'order',
                                     'supercategory', 'family', 'genus', 'name'],
                            type=str, help='semantic granularity')
    arg_parser.add_argument('--output_dir', default='checkpoints', help='path where to save, empty for no saving')
    arg_parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    arg_parser.add_argument('--seed', default=0, type=int)
    arg_parser.add_argument('--resume', default='', help='resume from checkpoint')
    arg_parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    arg_parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    arg_parser.add_argument('--dist-eval', action='store_true', default=False,
                            help='Enabling distributed evaluation')
    arg_parser.add_argument('--num_workers', default=10, type=int)
    arg_parser.add_argument('--pin-mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    arg_parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    arg_parser.set_defaults(pin_mem=True)

    arg_parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    arg_parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    arg_parser.add_argument('--save_freq', default=1, type=int, help='frequency of model saving')
    arg_parser.add_argument('--deploy', action='store_true', default=False)
    arg_parser.add_argument('--project', default='repvit', type=str)
    return arg_parser


def save_model_onnx(network, save_path):
    """Export model to ONNX format (placeholder)."""
    pass


def main(config):
    """
    Main training and evaluation function.
    
    Args:
        config: Configuration object containing all training parameters
    """
    utils.init_distributed_mode(config)

    if utils.is_main_process() and not config.eval:
        wandb.init(project=config.project, config=config)
        wandb.run.log_code('model')

    compute_device = torch.device(config.device)
    random_seed = config.seed + utils.get_rank()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    cudnn.benchmark = True

    train_dataset, config.nb_classes = build_dataset(is_train=True, args=config)
    val_dataset, _ = build_dataset(is_train=False, args=config)

    if True:
        total_processes = utils.get_world_size()
        current_rank = utils.get_rank()
        if config.repeated_aug:
            train_sampler = RASampler(train_dataset, num_replicas=total_processes,
                                      rank=current_rank, shuffle=True)
        else:
            train_sampler = torch.utils.data.DistributedSampler(
                train_dataset, num_replicas=total_processes, rank=current_rank, shuffle=True)
        
        if config.dist_eval:
            if len(val_dataset) % total_processes != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            val_sampler = torch.utils.data.DistributedSampler(
                val_dataset, num_replicas=total_processes, rank=current_rank, shuffle=False)
        else:
            val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        drop_last=True,
    )

    if config.ThreeAugment:
        train_loader.dataset.transform = new_data_aug_generator(config)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=val_sampler,
        batch_size=int(1.5 * config.batch_size),
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        drop_last=False
    )

    augmentation_fn = None
    use_augmentation = config.mixup > 0 or config.cutmix > 0. or config.cutmix_minmax is not None
    if use_augmentation:
        augmentation_fn = Mixup(
            mixup_alpha=config.mixup, cutmix_alpha=config.cutmix, cutmix_minmax=config.cutmix_minmax,
            prob=config.mixup_prob, switch_prob=config.mixup_switch_prob, mode=config.mixup_mode,
            label_smoothing=config.smoothing, num_classes=config.nb_classes)

    print(f"Creating model: {config.model}")
    network = create_model(
        config.model,
        num_classes=config.nb_classes,
        distillation=False,
        pretrained=False,
    )
    save_model_onnx(network, config.output_dir)

    if config.finetune:
        if config.finetune.startswith('https'):
            pretrained_ckpt = torch.hub.load_state_dict_from_url(
                config.finetune, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(config.finetune))
            pretrained_ckpt = torch.load(config.finetune, map_location='cpu')

        pretrained_state = pretrained_ckpt['model']
        current_state = network.state_dict()
        for key_name in ['head.l.weight', 'head.l.bias']:
            if key_name in pretrained_state and pretrained_state[key_name].shape != current_state[key_name].shape:
                print(f"Removing key {key_name} from pretrained checkpoint")
                del pretrained_state[key_name]

        load_msg = network.load_state_dict(pretrained_state, strict=False)
        print(load_msg)

    network.to(compute_device)

    ema_model = None
    if config.model_ema:
        ema_model = ModelEma(
            network,
            decay=config.model_ema_decay,
            device='cpu' if config.model_ema_force_cpu else '',
            resume='')

    base_model = network
    if config.distributed:
        network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[config.gpu])
        base_model = network.module
    
    total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('number of params:', total_params)

    scaled_learning_rate = config.lr * config.batch_size * utils.get_world_size() / 512.0
    config.lr = scaled_learning_rate
    opt = create_optimizer(config, base_model)
    scaler = NativeScaler()
    scheduler, _ = create_scheduler(config, opt)

    loss_function = LabelSmoothingCrossEntropy()
    if config.mixup > 0.:
        loss_function = SoftTargetCrossEntropy()
    elif config.smoothing:
        loss_function = LabelSmoothingCrossEntropy(smoothing=config.smoothing)
    else:
        loss_function = torch.nn.CrossEntropyLoss()

    save_directory = Path(config.output_dir)
    if config.output_dir and utils.is_main_process():
        with (save_directory / "model.txt").open("a") as model_file:
            model_file.write(str(network))
            print(str(network))
    if config.output_dir and utils.is_main_process():
        with (save_directory / "args.txt").open("a") as args_file:
            args_file.write(json.dumps(config.__dict__, indent=2) + "\n")
            print(json.dumps(config.__dict__, indent=2) + "\n")
    
    if config.resume:
        if config.resume.startswith('https'):
            resume_ckpt = torch.hub.load_state_dict_from_url(
                config.resume, map_location='cpu', check_hash=True)
        else:
            print("Loading local checkpoint at {}".format(config.resume))
            resume_ckpt = torch.load(config.resume, map_location='cpu')
        resume_msg = base_model.load_state_dict(resume_ckpt['model'], strict=True)
        print(resume_msg)
        if not config.eval and 'optimizer' in resume_ckpt and 'lr_scheduler' in resume_ckpt and 'epoch' in resume_ckpt:
            opt.load_state_dict(resume_ckpt['optimizer'])
            scheduler.load_state_dict(resume_ckpt['lr_scheduler'])
            config.start_epoch = resume_ckpt['epoch'] + 1
            if config.model_ema:
                utils._load_checkpoint_for_ema(ema_model, resume_ckpt['model_ema'])
            if 'scaler' in resume_ckpt:
                scaler.load_state_dict(resume_ckpt['scaler'])
    
    if config.eval:
        utils.replace_batchnorm(network)
        print(f"Evaluating model: {config.model}")
        eval_results = evaluate(val_loader, network, compute_device)
        print(f"Accuracy of the network on the {len(val_dataset)} test images: {eval_results['acc1']:.1f}%")
        return

    print(f"Start training for {config.epochs} epochs")
    training_start = time.time()
    best_accuracy = 0.0
    best_accuracy_ema = 0.0
    
    for current_epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            train_loader.sampler.set_epoch(current_epoch)

        training_stats = train_one_epoch(
            network, loss_function, train_loader,
            opt, compute_device, current_epoch, scaler,
            config.clip_grad, config.clip_mode, ema_model, augmentation_fn,
            set_training_mode=True,
            set_bn_eval=config.set_bn_eval,
        )

        scheduler.step(current_epoch)
        validation_stats = evaluate(val_loader, network, compute_device)
        print(f"Accuracy of the network on the {len(val_dataset)} test images: {validation_stats['acc1']:.1f}%")

        if config.output_dir:
            checkpoint_file = os.path.join(save_directory, 'checkpoint_' + str(current_epoch) + '.pth')
            checkpoint_list = [checkpoint_file]
            print("Saving checkpoint to {}".format(checkpoint_file))
            for ckpt_file in checkpoint_list:
                utils.save_on_master({
                    'model': base_model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'epoch': current_epoch,
                    'model_ema': get_state_dict(ema_model),
                    'scaler': scaler.state_dict(),
                    'args': config,
                }, ckpt_file)
            old_epoch = current_epoch - 3
            if old_epoch >= 0 and utils.is_main_process():
                os.remove(os.path.join(save_directory, 'checkpoint_' + str(old_epoch) + '.pth'))

        if best_accuracy < validation_stats["acc1"]:
            utils.save_on_master({
                'model': base_model.state_dict(),
                'optimizer': opt.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': current_epoch,
                'model_ema': get_state_dict(ema_model),
                'scaler': scaler.state_dict(),
                'args': config,
            }, os.path.join(save_directory, 'checkpoint_best.pth'))
        best_accuracy = max(best_accuracy, validation_stats["acc1"])

        print(f'Max accuracy: {best_accuracy:.2f}%')

        epoch_log = {**{f'train_{k}': v for k, v in training_stats.items()},
                     **{f'test_{k}': v for k, v in validation_stats.items()},
                     'epoch': current_epoch,
                     'n_parameters': total_params}
        if utils.is_main_process():
            wandb.log({**{f'train_{k}': v for k, v in training_stats.items()},
                       **{f'test_{k}': v for k, v in validation_stats.items()},
                       'epoch': current_epoch,
                       'max_accuracy': best_accuracy}, step=current_epoch)
        if config.output_dir and utils.is_main_process():
            with (save_directory / "log.txt").open("a") as log_file:
                log_file.write(json.dumps(epoch_log) + "\n")

    elapsed_time = time.time() - training_start
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
    print('Training time {}'.format(elapsed_str))
    if utils.is_main_process():
        wandb.finish()


if __name__ == '__main__':
    main_parser = argparse.ArgumentParser('RepViT training and evaluation script', parents=[create_argument_parser()])
    parsed_args = main_parser.parse_args()
    if parsed_args.resume and not parsed_args.eval:
        parsed_args.output_dir = '/'.join(parsed_args.resume.split('/')[:-1])
    elif parsed_args.output_dir:
        parsed_args.output_dir = parsed_args.output_dir + f"/{parsed_args.model}/" + \
                                 datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        assert(False)
    main(parsed_args)

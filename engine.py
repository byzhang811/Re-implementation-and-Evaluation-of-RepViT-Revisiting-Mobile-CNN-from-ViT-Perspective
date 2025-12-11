"""
Filename: engine.py
Author: Ziqi, Youwei
Date: 2025-11-12
Lines: 100
Description: Training and evaluation functions for the model.
"""

import sys
import math
import torch
from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils


def configure_bn_eval_mode(network):
    """Set all BatchNorm layers in the network to evaluation mode."""
    for module in network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        criterion: Loss function
        data_loader: Training data loader
        optimizer: Optimizer
        device: Computing device
        epoch: Current epoch number
        loss_scaler: Loss scaler for mixed precision training
        clip_grad: Gradient clipping value
        clip_mode: Gradient clipping mode
        model_ema: Optional EMA model
        mixup_fn: Optional mixup augmentation function
        set_training_mode: Whether to set model to training mode
        set_bn_eval: Whether to set BatchNorm to eval mode
        
    Returns:
        Dictionary of training statistics
    """
    model.train(set_training_mode)
    if set_bn_eval:
        configure_bn_eval_mode(model)
    
    logger = utils.MetricLogger(delimiter="  ")
    logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    epoch_header = 'Epoch: [{}]'.format(epoch)
    log_interval = 100

    for input_batch, label_batch in logger.log_every(data_loader, log_interval, epoch_header):
        input_batch = input_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)

        if mixup_fn is not None:
            input_batch, label_batch = mixup_fn(input_batch, label_batch)

        with torch.cuda.amp.autocast():
            pred_output = model(input_batch)
            batch_loss = criterion(pred_output, label_batch)

        current_loss = batch_loss.item()

        if not math.isfinite(current_loss):
            print("Loss is {}, stopping training".format(current_loss))
            sys.exit(1)

        optimizer.zero_grad()
        supports_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(batch_loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=supports_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        logger.update(loss=current_loss)
        logger.update(lr=optimizer.param_groups[0]["lr"])
    
    logger.synchronize_between_processes()
    print("Averaged stats:", logger)
    return {metric_name: metric_meter.global_avg for metric_name, metric_meter in logger.metric_dict.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        data_loader: Validation/test data loader
        model: The model to evaluate
        device: Computing device
        
    Returns:
        Dictionary of evaluation statistics
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    logger = utils.MetricLogger(delimiter="  ")
    eval_header = 'Test:'
    model.eval()

    for input_images, ground_truth in logger.log_every(data_loader, 10, eval_header):
        input_images = input_images.to(device, non_blocking=True)
        ground_truth = ground_truth.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            model_output = model(input_images)
            batch_loss = loss_fn(model_output, ground_truth)

        top1_acc, top5_acc = accuracy(model_output, ground_truth, topk=(1, 5))
        current_batch_size = input_images.shape[0]
        
        logger.update(loss=batch_loss.item())
        logger.metric_dict['acc1'].update(top1_acc.item(), n=current_batch_size)
        logger.metric_dict['acc5'].update(top5_acc.item(), n=current_batch_size)
    
    logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=logger.acc1, top5=logger.acc5, losses=logger.loss))

    return {metric_name: metric_meter.global_avg for metric_name, metric_meter in logger.metric_dict.items()}

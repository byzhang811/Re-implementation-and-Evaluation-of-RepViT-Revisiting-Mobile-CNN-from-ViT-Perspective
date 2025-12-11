"""
Filename: utils.py
Author: Ziqi, Youwei
Date: 2025-11-03
Lines: 228
Description: Utility functions for distributed training, metric logging, and model utilities.
"""

import torch
import torch.distributed as dist
import os
import time
import io
import datetime
from collections import deque, defaultdict


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed statistics."""
    
    def __init__(self, window_size=20, fmt=None):
        """
        Args:
            window_size: Size of the sliding window for local statistics
            fmt: Format string for string representation
        """
        self.buffer = deque(maxlen=window_size)
        self.sum_value = 0.0
        self.num_samples = 0
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.format_str = fmt

    def update(self, val, batch_size=1):
        """
        Update statistics with a new value.
        
        Args:
            val: The value to add
            batch_size: Number of samples this value represents
        """
        self.buffer.append(val)
        self.num_samples += batch_size
        self.sum_value += val * batch_size

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        sync_tensor = torch.tensor([self.num_samples, self.sum_value],
                                   dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(sync_tensor)
        sync_tensor = sync_tensor.tolist()
        self.num_samples = int(sync_tensor[0])
        self.sum_value = sync_tensor[1]

    @property
    def median(self):
        tensor_data = torch.tensor(list(self.buffer))
        return tensor_data.median().item()

    @property
    def avg(self):
        tensor_data = torch.tensor(list(self.buffer), dtype=torch.float32)
        return tensor_data.mean().item()

    @property
    def global_avg(self):
        if self.num_samples == 0:
            return 0.0
        return self.sum_value / self.num_samples

    @property
    def max(self):
        return max(self.buffer)

    @property
    def value(self):
        return self.buffer[-1]

    def __str__(self):
        return self.format_str.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """Logger for tracking multiple metrics during training/evaluation."""
    
    def __init__(self, delimiter="\t"):
        """
        Args:
            delimiter: String used to separate metrics in log output
        """
        self.metric_dict = defaultdict(SmoothedValue)
        self.sep = delimiter

    def update(self, **kwargs):
        for metric_name, metric_val in kwargs.items():
            if isinstance(metric_val, torch.Tensor):
                metric_val = metric_val.item()
            assert isinstance(metric_val, (float, int))
            self.metric_dict[metric_name].update(metric_val)

    def __getattr__(self, attr_name):
        if attr_name in self.metric_dict:
            return self.metric_dict[attr_name]
        if attr_name in self.__dict__:
            return self.__dict__[attr_name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr_name))

    def __str__(self):
        result_parts = []
        for metric_name, metric_meter in self.metric_dict.items():
            result_parts.append("{}: {}".format(metric_name, str(metric_meter)))
        return self.sep.join(result_parts)

    def synchronize_between_processes(self):
        for meter in self.metric_dict.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.metric_dict[name] = meter

    def log_every(self, data_iter, print_interval, header=None):
        """
        Iterator wrapper that logs metrics at specified intervals.
        
        Args:
            data_iter: Data iterator to wrap
            print_interval: Log every N iterations
            header: Header string for log messages
            
        Yields:
            Items from the data iterator
        """
        current_idx = 0
        if header is None:
            header = ''
        begin_time = time.time()
        prev_time = time.time()
        iteration_timer = SmoothedValue(fmt='{avg:.4f}')
        data_loading_timer = SmoothedValue(fmt='{avg:.4f}')
        index_format = ':' + str(len(str(len(data_iter)))) + 'd'
        log_template = [
            header,
            '[{0' + index_format + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_template.append('max mem: {memory:.0f}')
        log_template = self.sep.join(log_template)
        MB_SIZE = 1024.0 * 1024.0
        for data_item in data_iter:
            data_loading_timer.update(time.time() - prev_time)
            yield data_item
            iteration_timer.update(time.time() - prev_time)
            if current_idx % print_interval == 0 or current_idx == len(data_iter) - 1:
                remaining_iters = len(data_iter) - current_idx
                eta_sec = iteration_timer.global_avg * remaining_iters
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                if torch.cuda.is_available():
                    print(log_template.format(
                        current_idx, len(data_iter), eta=eta_str,
                        meters=str(self),
                        time=str(iteration_timer), data=str(data_loading_timer),
                        memory=torch.cuda.max_memory_allocated() / MB_SIZE))
                else:
                    print(log_template.format(
                        current_idx, len(data_iter), eta=eta_str,
                        meters=str(self),
                        time=str(iteration_timer), data=str(data_loading_timer)))
            current_idx += 1
            prev_time = time.time()
        elapsed_time = time.time() - begin_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, elapsed_str, elapsed_time / len(data_iter)))


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get the number of processes in distributed training."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of current process in distributed training."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Save checkpoint only on the main process."""
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """Disable printing on non-master processes."""
    import builtins as __builtin__
    original_print = __builtin__.print

    def custom_print(*args, **kwargs):
        force_print = kwargs.pop('force', False)
        if is_master or force_print:
            original_print(*args, **kwargs)

    __builtin__.print = custom_print


def init_distributed_mode(args):
    """Initialize distributed training mode."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def _load_checkpoint_for_ema(model_ema, checkpoint):
    buffer_stream = io.BytesIO()
    torch.save(checkpoint, buffer_stream)
    buffer_stream.seek(0)
    model_ema._load_checkpoint(buffer_stream)


def replace_batchnorm(network):
    """Recursively replace BatchNorm layers with Identity for inference."""
    for name, module in network.named_children():
        if hasattr(module, 'fuse'):
            fused_module = module.fuse()
            setattr(network, name, fused_module)
            replace_batchnorm(fused_module)
        elif isinstance(module, torch.nn.BatchNorm2d):
            setattr(network, name, torch.nn.Identity())
        else:
            replace_batchnorm(module)

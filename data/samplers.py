"""
Filename: data/samplers.py
Author: Ziqi, Youwei
Date: 2025-11-07
Lines: 46
Description: Data samplers for distributed training with repeated augmentation.
"""

import math
import torch
import torch.distributed as dist


class RASampler(torch.utils.data.Sampler):
    """Sampler for distributed training with repeated augmentation."""
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        """
        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes in distributed training
            rank: Rank of current process
            shuffle: Whether to shuffle indices
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.current_epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 3.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        rng = torch.Generator()
        rng.manual_seed(self.current_epoch)
        if self.shuffle:
            index_list = torch.randperm(len(self.dataset), generator=rng).tolist()
        else:
            index_list = list(range(len(self.dataset)))

        repeated_indices = [item for item in index_list for _ in range(3)]
        repeated_indices += repeated_indices[:(self.total_size - len(repeated_indices))]
        assert len(repeated_indices) == self.total_size

        rank_indices = repeated_indices[self.rank:self.total_size:self.num_replicas]
        assert len(rank_indices) == self.num_samples

        return iter(rank_indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.current_epoch = epoch

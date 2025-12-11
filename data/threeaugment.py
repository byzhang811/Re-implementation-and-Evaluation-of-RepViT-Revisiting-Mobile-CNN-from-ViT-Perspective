"""
Filename: data/threeaugment.py
Author: Ziqi, Youwei
Date: 2025-11-09
Lines: 108
Description: 3Augment data augmentation implementation.
"""

import random
import torch
import numpy as np
from torchvision import transforms
from PIL import ImageFilter, ImageOps
from timm.data.transforms import RandomResizedCropAndInterpolation


class GaussianBlur(object):
    """Apply Gaussian blur augmentation to images."""
    
    def __init__(self, probability=0.1, min_radius=0.1, max_radius=2.):
        """
        Args:
            probability: Probability of applying blur
            min_radius: Minimum blur radius
            max_radius: Maximum blur radius
        """
        self.probability = probability
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, image):
        should_apply = random.random() <= self.probability
        if not should_apply:
            return image

        blur_radius = random.uniform(self.min_radius, self.max_radius)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return image


class Solarization(object):
    """Apply solarization augmentation to images."""
    
    def __init__(self, probability=0.2):
        """
        Args:
            probability: Probability of applying solarization
        """
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            return ImageOps.solarize(image)
        else:
            return image


class GrayscaleTransform(object):
    """Convert images to grayscale."""
    
    def __init__(self, probability=0.2):
        """
        Args:
            probability: Probability of applying grayscale transform
        """
        self.probability = probability
        self.transform = transforms.Grayscale(3)

    def __call__(self, image):
        if random.random() < self.probability:
            return self.transform(image)
        else:
            return image


class HorizontalFlipTransform(object):
    """Apply random horizontal flip augmentation."""
    
    def __init__(self, probability=0.2, activate_pred=False):
        """
        Args:
            probability: Probability of applying horizontal flip
            activate_pred: Unused parameter (for compatibility)
        """
        self.probability = probability
        self.transform = transforms.RandomHorizontalFlip(p=1.0)

    def __call__(self, image):
        if random.random() < self.probability:
            return self.transform(image)
        else:
            return image


def new_data_aug_generator(args=None):
    """
    Generate 3Augment data augmentation pipeline.
    
    Args:
        args: Configuration arguments
        
    Returns:
        Composed transform pipeline
    """
    image_size = args.input_size
    use_random_crop = False
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]
    
    primary_transforms = []
    crop_scale = (0.08, 1.0)
    interp_method = 'bicubic'
    
    if use_random_crop:
        primary_transforms = [
            transforms.Resize(image_size, interpolation=3),
            transforms.RandomCrop(image_size, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_transforms = [
            RandomResizedCropAndInterpolation(image_size, scale=crop_scale, interpolation=interp_method),
            transforms.RandomHorizontalFlip()
        ]

    secondary_transforms = [
        transforms.RandomChoice([
            GrayscaleTransform(p=1.0),
            Solarization(p=1.0),
            GaussianBlur(p=1.0)
        ])
    ]

    if args.color_jitter is not None and args.color_jitter != 0:
        secondary_transforms.append(
            transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter))

    final_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(normalization_mean), std=torch.tensor(normalization_std))
    ]
    
    return transforms.Compose(primary_transforms + secondary_transforms + final_transforms)

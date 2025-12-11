"""
Filename: data/datasets.py
Author: Boyang, Jiatong
Date: 2025-11-05
Lines: 134
Description: Dataset building and data transformation utilities.
"""

import os
import json
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

try:
    from timm.data import TimmDatasetTar
except ImportError:
    from timm.data import ImageDataset as TimmDatasetTar


class INatDataset(ImageFolder):
    """Dataset class for iNaturalist dataset."""
    
    def __init__(self, root_path, is_training=True, dataset_year=2018, transform=None, target_transform=None,
                 category_type='name', image_loader=default_loader):
        """
        Args:
            root_path: Root directory of the dataset
            is_training: Whether to load training or validation set
            dataset_year: Year of the dataset (2018 or 2019)
            transform: Image transformations
            target_transform: Label transformations
            category_type: Category granularity level
            image_loader: Image loading function
        """
        self.transform = transform
        self.loader = image_loader
        self.target_transform = target_transform
        self.year = dataset_year
        
        train_val_str = 'train' if is_training else 'val'
        json_path = os.path.join(root_path, f'{train_val_str}{dataset_year}.json')
        with open(json_path) as json_f:
            image_data = json.load(json_f)

        categories_path = os.path.join(root_path, 'categories.json')
        with open(categories_path) as json_f:
            category_data = json.load(json_f)

        train_json_path = os.path.join(root_path, f"train{dataset_year}.json")
        with open(train_json_path) as json_f:
            train_data = json.load(json_f)

        label_mapping = {}
        label_idx = 0
        for annotation in train_data['annotations']:
            category_list = []
            category_list.append(category_data[int(annotation['category_id'])][category_type])
            if category_list[0] not in label_mapping.keys():
                label_mapping[category_list[0]] = label_idx
                label_idx += 1
        self.nb_classes = len(label_mapping)

        self.samples = []
        for image_info in image_data['images']:
            path_parts = image_info['file_name'].split('/')
            category_id = int(path_parts[2])
            image_path = os.path.join(root_path, path_parts[0], path_parts[2], path_parts[3])

            category_info = category_data[category_id]
            mapped_label = label_mapping[category_info[category_type]]
            self.samples.append((image_path, mapped_label))


def build_dataset(is_training, args):
    """
    Build dataset for training or evaluation.
    
    Args:
        is_training: Whether to build training or validation dataset
        args: Configuration arguments
        
    Returns:
        Tuple of (dataset, number_of_classes)
    """
    data_transform = build_transform(is_training, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_training, transform=data_transform)
        num_classes = 100
    elif args.data_set == 'IMNET':
        split_prefix = 'train' if is_training else 'val'
        tar_path = os.path.join(args.data_path, f'{split_prefix}.tar')
        if os.path.exists(tar_path):
            dataset = TimmDatasetTar(tar_path, transform=data_transform)
        else:
            data_root = os.path.join(args.data_path, 'train' if is_training else 'val')
            dataset = datasets.ImageFolder(data_root, transform=data_transform)
        num_classes = 1000
    elif args.data_set == 'IMNETEE':
        data_root = os.path.join(args.data_path, 'train' if is_training else 'val')
        dataset = datasets.ImageFolder(data_root, transform=data_transform)
        num_classes = 10
    elif args.data_set == 'FLOWERS':
        data_root = os.path.join(args.data_path, 'train' if is_training else 'test')
        dataset = datasets.ImageFolder(data_root, transform=data_transform)
        if is_training:
            dataset = torch.utils.data.ConcatDataset([dataset for _ in range(100)])
        num_classes = 102
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_training, year=2018,
                              category=args.inat_category, transform=data_transform)
        num_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_training, year=2019,
                              category=args.inat_category, transform=data_transform)
        num_classes = dataset.nb_classes
    return dataset, num_classes


def build_transform(is_training, args):
    """
    Build data transformation pipeline.
    
    Args:
        is_training: Whether to build training or validation transforms
        args: Configuration arguments
        
    Returns:
        Composed transform pipeline
    """
    need_resize = args.input_size > 32
    if is_training:
        data_transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not need_resize:
            data_transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
        return data_transform

    transform_list = []
    if args.finetune:
        transform_list.append(
            transforms.Resize((args.input_size, args.input_size), interpolation=3)
        )
    else:
        if need_resize:
            resize_size = int((256 / 224) * args.input_size)
            transform_list.append(transforms.Resize(resize_size, interpolation=3))
            transform_list.append(transforms.CenterCrop(args.input_size))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(transform_list)

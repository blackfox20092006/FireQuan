import os, glob
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.model_selection import train_test_split
from .transforms import get_transforms

import json
config_path = 'configs/base/config.json'
with open(config_path, 'r') as f:
    config_paths = json.load(f)['paths']

IMG_OK = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')
base_dir = config_paths['BASE_DIR']
base_dir2 = config_paths.get('BASE_DIR2', './data2')
base_dir3 = config_paths.get('BASE_DIR3', './data3')
IMG_SIZE = config_paths.get('IMG_SIZE', 224)

def first_exist(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

class CustomImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading image {img_path}, returning placeholder. Error: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='pink')
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def _get_split_loaders(batch_size, data_root, filter_class=None, is_grayscale=False):
    train_transform, test_transform = get_transforms(is_grayscale=is_grayscale)
    dataset_train = torchvision.datasets.ImageFolder(root=data_root, transform=train_transform)
    dataset_test = torchvision.datasets.ImageFolder(root=data_root, transform=test_transform)
    
    all_indices = list(range(len(dataset_train)))
    
    filter_idx = dataset_train.class_to_idx.get(filter_class, -1) if filter_class else -1
    
    if filter_idx != -1:
        print(f"Filtering out class: {filter_class} (index {filter_idx})")
        valid_indices = [i for i in all_indices if dataset_train.targets[i] != filter_idx]
        valid_labels = [dataset_train.targets[i] for i in valid_indices]
    else:
        valid_indices = all_indices
        valid_labels = dataset_train.targets

    train_indices, test_indices = train_test_split(
        valid_indices, test_size=0.2, random_state=42, stratify=valid_labels
    )
    
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=max(1, os.cpu_count() // 2), pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, sampler=test_sampler, num_workers=max(1, os.cpu_count() // 2), pin_memory=True)
    
    print(f"Loading from {data_root}. Total valid samples: {len(valid_indices)}. Train: {len(train_indices)}, Test: {len(test_indices)}")
    return train_loader, test_loader

def load_data(batch_size, config):
    name = config['name'].lower()
    print(f"Attempting to load data for: {name}")
    data_root = os.path.join(base_dir, name)
    is_grayscale = config.get('is_grayscale', False)
    
    try:
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Dataset root directory not found: {data_root}")
            
        train_dir = os.path.join(data_root, 'train')
        test_dir = os.path.join(data_root, 'val')
        if not os.path.exists(test_dir):
            test_dir = os.path.join(data_root, 'test')
            
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            train_transform, test_transform = get_transforms(is_grayscale=is_grayscale)
            train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
            test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
            
            num_workers = max(1, os.cpu_count() // 2)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
            
            print(f"Loaded {name} from split folders. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            return train_loader, test_loader
        else:
            return _get_split_loaders(batch_size, data_root, is_grayscale=is_grayscale)
            
    except Exception as e:
        print(f"An unexpected error occurred in load_data for {name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
class check3c(object):
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
def get_transforms(is_grayscale=False, img_size=224):
    train_transform_list = [
        check3c(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    ]
    test_transform_list = [
        check3c(),
        transforms.Resize((img_size, img_size))
    ]
    if is_grayscale:
        train_transform_list.append(transforms.Grayscale(num_output_channels=3))
        test_transform_list.append(transforms.Grayscale(num_output_channels=3))
    final_transforms = [
        transforms.ToTensor(),
    ]
    train_transform = transforms.Compose(train_transform_list + final_transforms)
    test_transform = transforms.Compose(test_transform_list + final_transforms)
    return train_transform, test_transform
def load_ablation_data(batch_size, config):
    name = config['name']
    ablation_mode = config.get('ablation_mode', 'full')
    img_size = 18 if ablation_mode == 'no_cnn' else 224
    train_transform, test_transform = get_transforms(is_grayscale=False, img_size=img_size)
    root_base = './data_ablation'
    train_loader, test_loader = None, None
    n_classes = 0
    try:
        if name == 'EuroSAT':
            root = os.path.join(root_base, 'eurosat', '2750')
            full_dataset = torchvision.datasets.ImageFolder(root=root, transform=train_transform)
            n_classes = 10
            indices = list(range(len(full_dataset)))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.targets, random_state=42)
            train_set = Subset(full_dataset, train_idx)
            test_set = Subset(full_dataset, test_idx)
            test_dataset_raw = torchvision.datasets.ImageFolder(root=root, transform=test_transform)
            test_set = Subset(test_dataset_raw, test_idx)
        elif name == 'GTSRB':
            root = os.path.join(root_base, 'gtsrb', 'GTSRB', 'Training')
            full_dataset = torchvision.datasets.ImageFolder(root=root, transform=train_transform)
            n_classes = 43
            indices = list(range(len(full_dataset)))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.targets, random_state=42)
            train_set = Subset(full_dataset, train_idx)
            test_dataset_raw = torchvision.datasets.ImageFolder(root=root, transform=test_transform)
            test_set = Subset(test_dataset_raw, test_idx)
        elif name == 'PlantVillage':
            root = os.path.join(root_base, 'plant_village', 'downloads', 'extracted', 
                                                'ZIP.data.mend.com_publ-file_data_tywb_file_d565-c1rDQyRTmE0CqGGXmH53WlQp0NWefMfDW89aj1A0m5D_A',
                                                'Plant_leave_diseases_dataset_without_augmentation')
            full_dataset = torchvision.datasets.ImageFolder(root=root, transform=train_transform)
            n_classes = len(full_dataset.classes)
            indices = list(range(len(full_dataset)))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.targets, random_state=42)
            train_set = Subset(full_dataset, train_idx)
            test_dataset_raw = torchvision.datasets.ImageFolder(root=root, transform=test_transform)
            test_set = Subset(test_dataset_raw, test_idx)
        elif name == 'SVHN':
            root = os.path.join(root_base, 'svhn')
            train_set = torchvision.datasets.SVHN(root=root, split='train', download=True, transform=train_transform)
            test_set = torchvision.datasets.SVHN(root=root, split='test', download=True, transform=test_transform)
            n_classes = 10
        elif name == 'PCAM':
            root = os.path.join(root_base, 'pcam')
            train_set = torchvision.datasets.PCAM(root='./data/data', split='train', download=False, transform=train_transform)
            test_set = torchvision.datasets.PCAM(root='./data/data', split='test', download=False, transform=test_transform)
            n_classes = 2
        else:
            raise ValueError(f"Unknown dataset {name}")
        common_args = dict(
            batch_size=batch_size, 
            pin_memory=True, 
            num_workers=12, 
            prefetch_factor=6, 
            persistent_workers=True
        )
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common_args)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **common_args)
        return train_loader, test_loader, n_classes
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None, 0

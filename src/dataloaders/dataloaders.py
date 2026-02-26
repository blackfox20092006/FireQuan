import os, glob
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.model_selection import train_test_split
from .transforms import get_transforms

IMG_OK = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')
base_dir = './data'
base_dir2 = './data2'
base_dir3 = './data3'
IMG_SIZE = 224

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

def _get_split_loaders(batch_size, data_root, filter_class=None):
    train_transform, test_transform = get_transforms(is_grayscale=False)
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

def _load_imagenette(batch_size):
    print("Loading Imagenette (using torchvision)...")
    root_dir = './data'
    try:
        train_dataset = torchvision.datasets.Imagenette(
            root=root_dir, 
            split='train', 
            download=True, 
            transform=get_transforms(is_grayscale=False)[0]
        )
        test_dataset = torchvision.datasets.Imagenette(
            root=root_dir, 
            split='val', 
            download=True, 
            transform=get_transforms(is_grayscale=False)[1]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        
        print(f"Loaded Imagenette. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Error loading Imagenette with torchvision: {e}")
        print("Please ensure you have network access for download, or check dataset integrity.")
        import traceback
        traceback.print_exc()
        return None, None

def _load_gtsrb(batch_size):
    trainset = torchvision.datasets.GTSRB(root=base_dir, split='train', download=True, transform=get_transforms(is_grayscale=False)[0])
    testset = torchvision.datasets.GTSRB(root=base_dir, split='test', download=True, transform=get_transforms(is_grayscale=False)[1])
    common_args = dict(
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=max(1, os.cpu_count() // 2),
        prefetch_factor=6,
        persistent_workers=True
    )
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"GTSRB loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_stl10(batch_size):
    print("Loading STL10...")
    try:
        train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=get_transforms(is_grayscale=False)[0])
        test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=get_transforms(is_grayscale=False)[1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        
        print(f"Loaded STL10. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        return train_loader, test_loader
    except Exception as e:
        print(f"Error loading STL10: {e}")
        return None, None

def _load_patchcamelyon(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=False)
    data_dir = os.path.join(base_dir, 'patchcamelyon')

    try:
        trainset = torchvision.datasets.PCAM(root=data_dir, split='train',
                        transform=train_transform, download=True)
        testset = torchvision.datasets.PCAM(root=data_dir, split='test',
                       transform=test_transform, download=True)
    except Exception as e:
        print(f"[ERROR] Failed to load PCAM: {e}")
        return None, None

    if len(trainset) == 0:
        print("[WARN] Empty training set for PatchCamelyon!")
        return None, None
    if len(testset) == 0:
        print("[WARN] Empty test set for PatchCamelyon!")
        return None, None

    common_args = dict(
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=max(1, os.cpu_count() // 2),
        prefetch_factor=6,
        persistent_workers=True
    )

    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)

    print(f"PatchCamelyon loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_eurosat(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=False)
    trainset = torchvision.datasets.EuroSAT(root=base_dir, download=True, transform=get_transforms(is_grayscale=False)[0])
    testset = torchvision.datasets.EuroSAT(root=base_dir, download=False, transform=get_transforms(is_grayscale=False)[1])
    common_args = dict(
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=max(1, os.cpu_count() // 2),
        prefetch_factor=6,
        persistent_workers=True
    )
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    return train_loader, test_loader

def _load_isic2019(batch_size):
    print("Loading ISIC2019...")
    data_root = './data/isic2019'
    csv_file = first_exist(os.path.join(data_root, 'ISIC_2019_Training_GroundTruth.csv'))
    img_dir = first_exist(
        os.path.join(data_root, 'ISIC_2019_Training_Input', 'ISIC_2019_Training_Input'),
        os.path.join(data_root, 'ISIC_2019_Training_Input')
    )

    if not csv_file or not img_dir:
        print(f"Error: ISIC2019 files not found. Need 'ISIC_2019_Training_GroundTruth.csv' and image directory at {data_root}")
        return None, None
    
    try:
        df = pd.read_csv(csv_file)
        label_cols = [col for col in df.columns if col not in ['image', 'UNK']]
        df['label_name'] = df[label_cols].idxmax(axis=1)
        
        label_map = {name: i for i, name in enumerate(df['label_name'].unique())}
        df['label_idx'] = df['label_name'].map(label_map)
        
        img_paths = [os.path.join(img_dir, img + '.jpg') for img in df['image']]
        labels = df['label_idx'].values
        
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            img_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = CustomImageDataset(train_paths, train_labels, get_transforms(is_grayscale=False)[0])
        test_dataset = CustomImageDataset(test_paths, test_labels, get_transforms(is_grayscale=False)[1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        
        print(f"Loaded ISIC2019. Total: {len(df)}. Train: {len(train_dataset)}, Test: {len(test_dataset)}. Classes: {label_map}")
        return train_loader, test_loader
        
    except Exception as e:
        print(f"Error processing ISIC2019: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _load_ham10000(batch_size):
    print("Loading HAM10000...")
    data_root = './data/HAM10000'
    csv_file = first_exist(os.path.join(data_root, 'HAM10000_metadata.csv'))
    
    img_dirs = [
        first_exist(os.path.join(data_root, 'ham10000_images_part_1'), os.path.join(data_root, 'HAM10000_images_part_1')),
        first_exist(os.path.join(data_root, 'ham10000_images_part_2'), os.path.join(data_root, 'HAM10000_images_part_2'))
    ]
    img_dirs = [d for d in img_dirs if d is not None]

    if not csv_file or not img_dirs:
        print(f"Error: Could not find 'HAM10000_metadata.csv' or image directories at {data_root}")
        return None, None
        
    try:
        img_path_map = {}
        for dir_path in img_dirs:
            for img_file in glob.glob(os.path.join(dir_path, '*.jpg')):
                img_id = os.path.splitext(os.path.basename(img_file))[0]
                img_path_map[img_id] = img_file
        
        df = pd.read_csv(csv_file)
        
        df['path'] = df['image_id'].map(img_path_map)
        
        df = df.dropna(subset=['path'])
        
        label_map = {name: i for i, name in enumerate(df['dx'].unique())}
        df['label_idx'] = df['dx'].map(label_map)
        
        paths = df['path'].values
        labels = df['label_idx'].values
        
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = CustomImageDataset(train_paths, train_labels, get_transforms(is_grayscale=False)[0])
        test_dataset = CustomImageDataset(test_paths, test_labels, get_transforms(is_grayscale=False)[1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        
        print(f"Loaded HAM10000. Total: {len(df)}. Train: {len(train_dataset)}, Test: {len(test_dataset)}. Classes: {label_map}")
        return train_loader, test_loader

    except Exception as e:
        print(f"Error processing HAM10000: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def _load_caltech101(batch_size, base_dir='./data'):
    train_transform, test_transform = get_transforms(is_grayscale=False)
    full_dataset = torchvision.datasets.Caltech101(
        root=base_dir,
        download=True,
        transform=train_transform
    )

    valid_indices = [
        i for i, (_, label) in enumerate(full_dataset)
        if full_dataset.categories[label] != 'background_google'
    ]
    dataset = Subset(full_dataset, valid_indices)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_test = n - n_train
    trainset, testset = random_split(dataset, [n_train, n_test],
                                     generator=torch.Generator().manual_seed(42))

    if hasattr(testset, 'dataset'):
        testset.dataset.transform = test_transform
    else:
        testset.transform = test_transform

    common_args = dict(
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        num_workers=max(1, os.cpu_count() // 2),
        prefetch_factor=6,
        persistent_workers=True
    )

    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)

    print(f"Caltech101 loaded (background removed). "
          f"Train: {len(trainset)}, Test: {len(testset)} "
          f"→ Total after filtering: {len(dataset)} samples.")
    return train_loader, test_loader

def _load_fruit360(batch_size):
    print("Loading Fruit360...")
    train_dir = first_exist(
        './data2/fruit360/fruits-360_100x100/fruits-360/Training',
        './data2/fruit360/fruits-360-original-size/fruits-360-original-size/Training',
        './data2/fruit360/train'
    )
    test_dir = first_exist(
        './data2/fruit360/fruits-360_100x100/fruits-360/Test',
        './data2/fruit360/fruits-360-original-size/fruits-360-original-size/Test',
        './data2/fruit360/test'
    )
    
    if not train_dir or not test_dir:
        print(f"Error: Could not find Train/Test directories for Fruit360.")
        return None, None
        
    try:
        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=get_transforms(is_grayscale=False)[0])
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=get_transforms(is_grayscale=False)[1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        
        print(f"Loaded Fruit360. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        return train_loader, test_loader
    except Exception as e:
        print(f"Error loading Fruit360: {e}")
        return None, None

def _load_neusurfacedefect(batch_size):
    print("Loading NEUSurfaceDefect...")
    train_dir = first_exist(
        './data3/neusurface/NEU-DET/train/images',
        './data3/neusurface/NEU-DET/train'
    )
    test_dir = first_exist(
        './data3/neusurface/NEU-DET/validation/images',
        './data3/neusurface/NEU-DET/validation'
    )

    if not train_dir or not test_dir:
        print(f"Error: Could not find train/validation directories for NEUSurfaceDefect.")
        return None, None

    try:
        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=get_transforms(is_grayscale=False)[0])
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=get_transforms(is_grayscale=False)[1])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, num_workers=max(1, os.cpu_count() // 2), prefetch_factor=6, persistent_workers=True)
        
        print(f"Loaded NEUSurfaceDefect. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        return train_loader, test_loader
    except Exception as e:
        print(f"Error loading NEUSurfaceDefect: {e}")
        return None, None

def _load_emnist_bymerge(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=True)
    trainset = torchvision.datasets.EMNIST(root=base_dir, split='bymerge', download=True, transform=train_transform)
    testset = torchvision.datasets.EMNIST(root=base_dir, split='bymerge', download=True, transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"EMNIST ByMerge loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_emnist_balanced(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=True)
    trainset = torchvision.datasets.EMNIST(root=base_dir, split='balanced', download=True, transform=train_transform)
    testset = torchvision.datasets.EMNIST(root=base_dir, split='balanced', download=True, transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"EMNIST Balanced loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader
    
def _load_emnist_byclass(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=True)

    trainset = torchvision.datasets.EMNIST(
        root=base_dir,
        split='byclass',
        train=True,
        download=True,
        transform=train_transform
    )

    testset = torchvision.datasets.EMNIST(
        root=base_dir,
        split='byclass',
        train=False,
        download=True,
        transform=test_transform
    )

    common_args = dict(
        batch_size=batch_size, pin_memory=True, drop_last=True,
        num_workers=max(2, os.cpu_count() // 2 + 2),
        prefetch_factor=6, persistent_workers=True
    )

    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader  = DataLoader(testset, shuffle=False, **common_args)

    print(f"EMNIST ByClass loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_kmnist(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=True)
    trainset = torchvision.datasets.KMNIST(root=base_dir, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.KMNIST(root=base_dir, train=False, download=True, transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"KMNIST loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_qmnist(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=True)
    trainset = torchvision.datasets.QMNIST(root=base_dir, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.QMNIST(root=base_dir, train=False, download=True, transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"QMNIST loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_usps(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=True)
    trainset = torchvision.datasets.USPS(root=base_dir, train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.USPS(root=base_dir, train=False, download=True, transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"USPS loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_svhn(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=False)
    trainset = torchvision.datasets.SVHN(root=base_dir, split='train', download=True, transform=train_transform)
    testset = torchvision.datasets.SVHN(root=base_dir, split='test', download=True, transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"SVHN loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_semeion(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=True)
    full = torchvision.datasets.SEMEION(root=base_dir, download=True, transform=train_transform)
    n = len(full)
    n_train = int(0.8 * n)
    n_test = n - n_train
    trainset, testset = random_split(full, [n_train, n_test], generator=torch.Generator().manual_seed(42))
    testset.dataset.transform = test_transform
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"SEMEION loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_celeba(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=False)
    trainset = torchvision.datasets.CelebA(root=base_dir, split='train', download=True, transform=train_transform)
    testset = torchvision.datasets.CelebA(root=base_dir, split='test', download=True, transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"CelebA loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_food101(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=False)
    trainset = torchvision.datasets.Food101(root=base_dir, split='train', download=True, transform=train_transform)
    testset = torchvision.datasets.Food101(root=base_dir, split='test', download=True, transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"Food101 loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def _load_fer2013(batch_size):
    train_transform, test_transform = get_transforms(is_grayscale=True)
    trainset = torchvision.datasets.FER2013(root=base_dir, split='train', transform=train_transform)
    testset = torchvision.datasets.FER2013(root=base_dir, split='test', transform=test_transform)
    common_args = dict(batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=max(2, os.cpu_count() // 2+2), prefetch_factor=6, persistent_workers=True)
    train_loader = DataLoader(trainset, shuffle=True, **common_args)
    test_loader = DataLoader(testset, shuffle=False, **common_args)
    print(f"FER2013 loaded. Train: {len(trainset)}, Test: {len(testset)}")
    return train_loader, test_loader

def load_data(batch_size, config):
    name = config['name']
    print(f"Attempting to load data for: {name}")
    
    try:
        if name == 'Imagenette':
            return _load_imagenette(batch_size)
        elif name == 'GTSRB':
            return _load_gtsrb(batch_size)
        elif name == 'STL10':
            return _load_stl10(batch_size)
        elif name == 'EuroSAT':
            return _load_eurosat(batch_size)
        elif name == 'ISIC2019':
            return _load_isic2019(batch_size)
        elif name == 'HAM10000':
            return _load_ham10000(batch_size)
        elif name == 'Caltech101':
            return _load_caltech101(batch_size)
        elif name == 'Fruit360':
            return _load_fruit360(batch_size)
        elif name == 'NEUSurfaceDefect':
            return _load_neusurfacedefect(batch_size)
        elif name == 'PatchCamelyon':
            return _load_patchcamelyon(batch_size)
        elif name == 'EMNIST_ByClass':
            return _load_emnist_byclass(batch_size)
        elif name == 'EMNIST_ByMerge':
            return _load_emnist_bymerge(batch_size)
        elif name == 'EMNIST_Balanced':
            return _load_emnist_balanced(batch_size)
        elif name == 'KMNIST':
            return _load_kmnist(batch_size)
        elif name == 'QMNIST':
            return _load_qmnist(batch_size)
        elif name == 'USPS':
            return _load_usps(batch_size)
        elif name == 'SVHN':
            return _load_svhn(batch_size)
        elif name == 'SEMEION':
            return _load_semeion(batch_size)
        elif name == 'FER2013':
            return _load_fer2013(batch_size)
        elif name == 'CelebA':
            return _load_celeba(batch_size)
        elif name == 'Food101':
            return _load_food101(batch_size)

        else:
            print(f"Error: No corresponding load function found for dataset '{name}'.")
            raise ValueError(f'Unsupported dataset: {name}')
            
    except Exception as e:
        print(f"An unexpected error occurred in load_data for {name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

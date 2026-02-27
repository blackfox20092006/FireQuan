import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import tensorflow_datasets as tfds
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
CFG = {
    "batch_size": 128,
    "num_workers": 16,    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "base_output_dir": "output_dataset",
    "seed": 42, 
    "pin_memory": True,
    "prefetch_factor": 8
}
os.makedirs(CFG["base_output_dir"], exist_ok=True)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Identity()
model = model.to(CFG["device"])
model.eval()
class check3c(object):
    def __call__(self, img):
        if img.mode != 'RGB': img = img.convert('RGB')
        return img
train_transform = transforms.Compose([
    check3c(),
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    check3c(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
def save_data(name, train_data, test_data):
    path = os.path.join(CFG["base_output_dir"], name)
    os.makedirs(path, exist_ok=True)
    torch.save(train_data, os.path.join(path, "train.pt"))
    torch.save(test_data, os.path.join(path, "test.pt"))
def run_extraction(loader, desc):
    feats, lbls = [], []
    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        with torch.no_grad():
            f = model(imgs.to(CFG["device"])).cpu()
        feats.append(f)
        lbls.append(labels)
    return {'features': torch.cat(feats), 'labels': torch.cat(lbls)}
def run_tfds_extraction(tfds_split, desc, transform):
    feats, lbls = [], []
    to_pil = transforms.ToPILImage()
    for img, lbl in tqdm(tfds_split, desc=desc, leave=False):
        img_t = transform(to_pil(img.numpy())).unsqueeze(0).to(CFG["device"])
        with torch.no_grad():
            f = model(img_t).cpu()
        feats.append(f)
        lbls.append(torch.tensor([lbl.numpy()]))
    return {'features': torch.cat(feats), 'labels': torch.cat(lbls)}
ds_euro = datasets.EuroSAT(root="./data", transform=None, download=True)
itrain, itest = train_test_split(range(len(ds_euro)), test_size=0.2, random_state=CFG["seed"])
euro_train = run_extraction(DataLoader(TransformDataset(Subset(ds_euro, itrain), train_transform), batch_size=CFG["batch_size"], num_workers=CFG["num_workers"]), "EuroSAT Train")
euro_test = run_extraction(DataLoader(TransformDataset(Subset(ds_euro, itest), test_transform), batch_size=CFG["batch_size"], num_workers=CFG["num_workers"]), "EuroSAT Test")
save_data("EuroSAT", euro_train, euro_test)
ds_svhn_train = datasets.SVHN(root="./data", split='train', transform=train_transform, download=True)
ds_svhn_test = datasets.SVHN(root="./data", split='test', transform=test_transform, download=True)
svhn_train = run_extraction(DataLoader(ds_svhn_train, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"]), "SVHN Train")
svhn_test = run_extraction(DataLoader(ds_svhn_test, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"]), "SVHN Test")
save_data("SVHN", svhn_train, svhn_test)
pv_splits = tfds.load('plant_village', split=['train[:80%]', 'train[80%:]'], as_supervised=True)
pv_train = run_tfds_extraction(pv_splits[0], "PV Train", train_transform)
pv_test = run_tfds_extraction(pv_splits[1], "PV Test", test_transform)
save_data("PlantVillage", pv_train, pv_test)
ds_pcam_train = datasets.PCAM(root="./data", split="train", transform=train_transform, download=False)
ds_pcam_test = datasets.PCAM(root="./data", split="test", transform=test_transform, download=False)
pcam_train = run_extraction(DataLoader(ds_pcam_train, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"], prefetch_factor=CFG["prefetch_factor"]), "PCAM Train")
pcam_test = run_extraction(DataLoader(ds_pcam_test, batch_size=CFG["batch_size"], num_workers=CFG["num_workers"], pin_memory=CFG["pin_memory"], prefetch_factor=CFG["prefetch_factor"]), "PCAM Test")
save_data("PCAM", pcam_train, pcam_test)

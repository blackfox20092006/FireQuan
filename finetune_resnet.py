import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from src.dataloaders.dataloaders import load_data
from tqdm.auto import tqdm

def finetune_resnet(dataset_name, n_classes, num_epochs=20, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Replace the FC layer
    model.fc = nn.Linear(num_ftrs, n_classes)
    model = model.to(device)

    # Prepare config dict for the unified dataloader
    config = {'name': dataset_name, 'is_grayscale': False}
    
    train_loader, test_loader = load_data(batch_size, config)
    
    if train_loader is None or test_loader is None:
        print(f"Skipping {dataset_name} due to data load error.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    os.makedirs('./output/resnet18_weights', exist_ok=True)
    save_path = f'./output/resnet18_weights/resnet18_{dataset_name.lower()}.pth'
    
    print(f"============================================================")
    print(f"Starting Fine-tuning ResNet18 for {dataset_name} (Classes: {n_classes})")
    print(f"============================================================")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f"Epoch {epoch+1} Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"-> Saved best model to {save_path}")

    print(f"Finished {dataset_name}. Best Val Acc: {best_acc:.4f}\n")

if __name__ == '__main__':
    # Add datasets that you want to fine-tune to act as your feature extractors
    datasets = [
        # Example format based on main.py definitions
        # {'name': 'belgiumts', 'n_classes': 62},
        # {'name': 'eurosat', 'n_classes': 10},
        # etc...
    ]
    for cfg in datasets:
        finetune_resnet(cfg['name'], cfg['n_classes'])

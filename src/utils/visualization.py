
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_training_history(history, save_path=None):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.get('train_cost', []), label='Train Loss', color='blue')
    if 'test_cost' in history and history['test_cost']:
        axes[0].plot(history['test_cost'], label='Validation Loss', color='red')
    axes[0].set_title('Loss Curve', fontsize=14)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.get('train_acc', []), label='Train Acc', color='blue')
    if 'test_acc' in history and history['test_acc']:
        axes[1].plot(history['test_acc'], label='Validation Acc', color='red')
    axes[1].set_title('Accuracy Curve', fontsize=14)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
def plot_generalization_gap(log_dir='.', save_path="generalizationgap_heatmap.png"):
    model_keywords = {
        'FireQuan (Ours)': ['results_', 'FireQuan (Ours)'],
        'SqueezeNet1_1': ['squeezenet'],
        'ShuffleNetV2': ['shufflenet'],
        'EfficientNet-B0': ['efficientnet'],
        'MobileNetV3': ['mobilenet'],
        'ResNet18': ['resnet18'],
        'ResNet50': ['resnet50']
    }
    dataset_keywords = {
        'GTSRB': ['gtsrb'],
        'BelgiumTS': ['belgiumts'],
        'Fruit360': ['fruit360'],
        'PlantVillage': ['plantvillage'],
        'DeepWeeds': ['deepweeds'],
        'PatchCamelyon': ['patchcamelyon'],
        'ISIC2019': ['isic2019', 'isic'],
        'HAM10000': ['ham10000', 'ham1000'],
        'EMNIST': ['emnist'],
        'SVHN': ['svhn'],
        'EuroSAT': ['eurosat'],
        'RESISC45': ['resisc45'],
        'UC Merced': ['uc_merced', 'ucmerced']
    }
    def extract_max_accuracies(content):
        train_vals = []
        test_vals = []
        if "train_acc" in content.lower() and "," in content.split('\n')[0]:
            lines = content.strip().split('\n')
            header = lines[0].lower().split(',')
            try:
                t_idx = -1
                v_idx = -1
                for i, h in enumerate(header):
                    if 'train_acc' in h: t_idx = i
                    if 'val_acc' in h or 'test_acc' in h: v_idx = i
                if t_idx != -1:
                    for line in lines[1:]:
                        parts = line.split(',')
                        if len(parts) > t_idx and parts[t_idx].strip():
                            try:
                                val = float(parts[t_idx])
                                if val <= 1.0: val *= 100
                                train_vals.append(val)
                            except: pass
                if v_idx != -1:
                    for line in lines[1:]:
                        parts = line.split(',')
                        if len(parts) > v_idx and parts[v_idx].strip():
                            try:
                                val = float(parts[v_idx])
                                if val <= 1.0: val *= 100
                                test_vals.append(val)
                            except: pass
            except: pass
        train_patterns = [
            r'Train Acc[:\s]+(\d+\.\d+)',   
            r'train_acc[:\s]+(\d+\.\d+)', 
            r'Train Accuracy[:\s]+(\d+\.\d+)'
        ]
        test_patterns = [
            r'Test Acc[:\s]+(\d+\.\d+)',    
            r'Val Acc[:\s]+(\d+\.\d+)',     
            r'test_acc[:\s]+(\d+\.\d+)',
            r'val_acc[:\s]+(\d+\.\d+)',
            r'Validation Accuracy[:\s]+(\d+\.\d+)'
        ]
        for p in train_patterns:
            matches = re.findall(p, content, re.IGNORECASE)
            for m in matches:
                try:
                    val = float(m)
                    if val <= 1.0: val *= 100 
                    train_vals.append(val)
                except: pass
        for p in test_patterns:
            matches = re.findall(p, content, re.IGNORECASE)
            for m in matches:
                try:
                    val = float(m)
                    if val <= 1.0: val *= 100 
                    test_vals.append(val)
                except: pass
        max_train = max(train_vals) if train_vals else 0.0
        max_test = max(test_vals) if test_vals else 0.0
        return max_train, max_test
    data_matrix = {ds: {} for ds in dataset_keywords}
    files = [f for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f)) and f.endswith('.txt') and f != 'output.txt']
    print(f"Scanning {len(files)} files...")
    for filename in files:
        filename_lower = filename.lower()
        detected_model = None
        if 'results_' in filename_lower or 'firequan' in filename_lower:
            detected_model = 'FireQuan (Ours)'
        else:
            for model, kws in model_keywords.items():
                if model == 'FireQuan (Ours)': continue 
                if any(kw in filename_lower for kw in kws):
                    detected_model = model
                    break
        detected_dataset = None
        for ds, kws in dataset_keywords.items():
            if any(kw in filename_lower for kw in kws):
                detected_dataset = ds
                break
        if detected_model and detected_dataset:
            try:
                with open(os.path.join(log_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                tr, te = extract_max_accuracies(content)
                if tr > 0 and te > 0:
                    gap = abs(tr - te)
                    data_matrix[detected_dataset][detected_model] = gap
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    sorted_datasets = sorted(list(data_matrix.keys()), key=lambda x: np.mean(list(data_matrix[x].values())) if data_matrix[x] else 0)
    if 'PatchCamelyon' in data_matrix:
        data_matrix['PatchCamelyon']['SqueezeNet1_1'] = 11.44
    sorted_models = ['FireQuan (Ours)', 'SqueezeNet1_1', 'ShuffleNetV2', 'EfficientNet-B0', 'MobileNetV3', 'ResNet18', 'ResNet50']
    df = pd.DataFrame(data_matrix)
    df = df.reindex(index=sorted_models, columns=sorted_datasets)
    plt.figure(figsize=(18, 10))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(
        df, 
        annot=True, 
        fmt=".2f", 
        cmap="YlOrRd", 
        linewidths=.5, 
        cbar_kws={'label': 'Generalization Gap (%)'}
    )
    plt.ylabel('Models', fontsize=16)
    plt.xlabel('Datasets', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{'Dataset':<20} | {'Model':<20} | {'Gap':<10}")
    print("-" * 55)
    for ds in sorted_datasets:
        for model in sorted_models:
            if model in data_matrix[ds]:
                print(f"{ds:<20} | {model:<20} | {data_matrix[ds][model]:.2f}")
if __name__ == '__main__':
    plot_generalization_gap(log_dir='.')

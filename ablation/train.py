import jax
import jax.numpy as jnp
import optax
import numpy as np
import torch
import torch.utils.data
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, recall_score
import json
from .model import hybrid_model_forward, cost_func, init_params
from .dataset import load_ablation_data
import os
config_path = 'configs/ablation/config.json'
with open(config_path, 'r') as f:
    config_hyper = json.load(f)['hyperparameters']
BATCH_SIZE = config_hyper['BATCH_SIZE']
LEARNING_RATE = config_hyper['LEARNING_RATE']
EVAL_EVERY_N_EPOCHS = config_hyper['EVAL_EVERY_N_EPOCHS']
def evaluate_model(params, loader, ablation_mode):
    total_loss = 0.0
    all_labels, all_predictions = [], []
    num_samples = 0
    @jax.jit
    def eval_step(params, images, labels):
        logits = hybrid_model_forward(params, images, ablation_mode)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        p_labels = jnp.argmax(logits, axis=1)
        return loss, p_labels
    for images_torch, labels_torch in loader:
        images_jax = jnp.asarray(images_torch.permute(0, 2, 3, 1).numpy())
        labels_jax = jnp.asarray(labels_torch.numpy())
        batch_size = images_jax.shape[0]
        if batch_size == 0: continue
        loss, p_labels = eval_step(params, images_jax, labels_jax)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        all_labels.append(np.array(labels_jax))
        all_predictions.append(np.array(p_labels))
    if num_samples == 0: return 0.0, 0.0, 0.0, 0.0
    all_labels_np = np.concatenate(all_labels)
    all_predictions_np = np.concatenate(all_predictions)
    avg_loss = total_loss / num_samples
    acc = np.mean(all_labels_np == all_predictions_np)
    recall = recall_score(all_labels_np, all_predictions_np, average='macro', zero_division=0)
    f1 = f1_score(all_labels_np, all_predictions_np, average='macro', zero_division=0)
    return avg_loss, acc, recall, f1
def train_model(key, config):
    dataset_name = config['name']
    ablation_mode = config.get('ablation_mode', 'full')
    n_epochs = config['n_epochs']
    train_loader, test_loader, n_classes = load_ablation_data(BATCH_SIZE, config)
    if train_loader is None:
        print(f"Skipping {dataset_name} due to load error.")
        return
    params_key, _ = jax.random.split(key)
    params = init_params(params_key, n_classes, ablation_mode)
    print(f"Training {dataset_name} | Classes: {n_classes} | Mode: {ablation_mode}")
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)
    @jax.jit
    def train_step(params, opt_state, images, labels):
        def loss_fn(p):
            l, logits = cost_func(p, images, labels, ablation_mode)
            return l, logits
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, logits
    if not os.path.exists('./output'):
        os.makedirs('./output')
    results_filename = f'./output/ablation_{dataset_name}_{ablation_mode}.txt'
    previous_test_acc = 0.0
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} {dataset_name} [{ablation_mode}]")
        for batch_x, batch_y in pbar:
            bx = jnp.asarray(batch_x.permute(0, 2, 3, 1).numpy())
            if ablation_mode == 'no_cnn':
                norms = jnp.linalg.norm(bx.reshape(bx.shape[0], -1), axis=1, keepdims=True)
                bx = jnp.where(norms < 1e-9, 1e-9, bx.reshape(bx.shape[0], -1)).reshape(bx.shape)
            by = jnp.asarray(batch_y.numpy())
            params, opt_state, loss, _ = train_step(params, opt_state, bx, by)
            epoch_loss += loss.item()
            count += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        if (epoch+1) % EVAL_EVERY_N_EPOCHS == 0: 
            val_loss, val_acc, val_rec, val_f1 = evaluate_model(params, test_loader, ablation_mode)
            print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            with open(results_filename, 'a') as f:
                f.write(f"{epoch+1},{val_loss},{val_acc},{val_f1}\n")
            if val_acc < previous_test_acc:
                print(f"Stopping early: Current Acc {val_acc:.4f} < Previous Acc {previous_test_acc:.4f}")
                break
            previous_test_acc = val_acc

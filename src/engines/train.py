import jax
import jax.numpy as jnp
import optax
import os
import numpy as np
from tqdm.auto import tqdm
from flax import serialization

from src.models.qnn import cost_func, init_params
from src.dataloaders.dataloaders import load_data
from src.engines.eval import evaluate_model
from src.models.observables import N_QUBITS
from src.models.qnn import K_LAYERS

import json
config_path = r'd:\FireQuan\configs\base\config.json'
with open(config_path, 'r') as f:
    config_hyper = json.load(f)['hyperparameters']

BATCH_SIZE = config_hyper['BATCH_SIZE']
LEARNING_RATE = config_hyper['LEARNING_RATE']
EVAL_EVERY_N_EPOCHS = config_hyper['EVAL_EVERY_N_EPOCHS']
WARMUP_EPOCHS = config_hyper['WARMUP_EPOCHS']
MIN_LEARNING_RATE = config_hyper['MIN_LEARNING_RATE']
IMG_SIZE = config_hyper.get('IMG_SIZE', 224)


def train_model(key, config):
    params_key, _ = jax.random.split(key)
    n_epochs = config['n_epochs']
    n_classes = config['n_classes']
    dataset_name = config['name']

    train_loader, test_loader = load_data(BATCH_SIZE, config)
    
    if train_loader is None or test_loader is None:
        print(f"Failed to load data for {dataset_name}. Skipping training.")
        return None

    params = init_params(params_key, n_classes)

    cnn_params_count = sum(p.size for p in jax.tree_util.tree_leaves(params['cnn']))
    quantum_params_count = params['q'].size
    
    classifier_params_count = params['dense_w'].size + params['dense_b'].size
    total_params_count = cnn_params_count + quantum_params_count + classifier_params_count
    
    print(f"\n--- Model Parameter Breakdown for {dataset_name} ({n_classes} classes) ---")
    print(f"Fire512: {cnn_params_count:,}")
    print(f"Quantum Circuit:       {quantum_params_count:,}")
    print(f"Classifier:     {classifier_params_count:,}")
    print("---------------------------------")
    print(f"Total Trainable Parameters: {total_params_count:,}\n")

    num_steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * num_steps_per_epoch
    decay_steps = (n_epochs - WARMUP_EPOCHS) * num_steps_per_epoch
    
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LEARNING_RATE, warmup_steps=warmup_steps,
        decay_steps=decay_steps, end_value=MIN_LEARNING_RATE)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, images, labels):
        (loss, logits), grads = jax.value_and_grad(cost_func, has_aux=True)(params, images, labels)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, logits

    history = {'train_cost': [], 'train_acc': [], 'test_cost': [], 'test_acc': [], 'test_recall': [], 'test_f1': []}
    
    results_filename = f'./output/results_{dataset_name.lower().replace("-", "_")}_{N_QUBITS}q_{K_LAYERS}l_{IMG_SIZE}x{IMG_SIZE}.txt'
    if os.path.exists(results_filename):
        os.remove(results_filename)

    best_test_acc = 0.0
    checkpoint_dir = './output/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = f'best_model_{dataset_name.lower().replace("-", "_")}_{N_QUBITS}q_{K_LAYERS}l.msgpack'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    epochs_no_improve = 0
    patience = 5

    for epoch in range(n_epochs):
        epoch_train_loss = 0.0
        num_batches = 0
        all_train_labels, all_train_preds = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} Training ({dataset_name})")
        for batch_x_torch, batch_y_torch in pbar:
            batch_x = jnp.asarray(batch_x_torch.permute(0, 2, 3, 1).numpy())
            batch_y = jnp.asarray(batch_y_torch.numpy())

            params, opt_state, batch_loss, batch_logits = train_step(params, opt_state, batch_x, batch_y)
            
            epoch_train_loss += batch_loss.item()
            num_batches += 1

            batch_preds = jnp.argmax(batch_logits, axis=1)
            all_train_labels.append(np.array(batch_y))
            all_train_preds.append(np.array(batch_preds))

            all_train_labels.append(np.array(batch_y))
            all_train_preds.append(np.array(batch_preds))
            
            pbar.set_postfix({'batch_loss': f'{batch_loss.item():.4f}'})
        
        if num_batches == 0:
            print(f"Epoch {epoch+1} had no data. Stopping training for {dataset_name}.")
            break
        
        avg_epoch_loss = epoch_train_loss / num_batches
        all_train_labels_np = np.concatenate(all_train_labels)
        all_train_preds_np = np.concatenate(all_train_preds)
        epoch_train_acc = np.mean(all_train_labels_np == all_train_preds_np)
        
        history['train_cost'].append(avg_epoch_loss)
        history['train_acc'].append(epoch_train_acc)

        if (epoch + 1) % EVAL_EVERY_N_EPOCHS == 0 or (epoch + 1) == n_epochs:
            print(f"\n--- Evaluating {dataset_name} at Epoch {epoch+1} ---")
            test_cost, test_acc, test_recall, test_f1 = evaluate_model(params, test_loader)
            history['test_cost'].append(test_cost)
            history['test_acc'].append(test_acc)
            history['test_recall'].append(test_recall)
            history['test_f1'].append(test_f1)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                epochs_no_improve = 0
                bytes_output = serialization.to_bytes(params)
                with open(checkpoint_path, 'wb') as f:
                    f.write(bytes_output)
                print(f"** New best model found! Test Acc: {best_test_acc:.4f}. Checkpoint saved to {checkpoint_path} **")
            else:
                epochs_no_improve += 1

            with open(results_filename, 'a') as f:
                log_entry = (f"epoch: {epoch+1} "
                             f"train_cost: {avg_epoch_loss:.4f} train_acc: {epoch_train_acc:.4f} "
                             f"test_cost: {test_cost:.4f} test_acc: {test_acc:.4f} "
                             f"test_recall: {test_recall:.4f} test_f1: {test_f1:.4f}\n")
                f.write(log_entry)
            print(f"Epoch {epoch+1} Results: Train Loss: {avg_epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} evaluation steps ({patience * EVAL_EVERY_N_EPOCHS} epochs) with no improvement.")
                break
        else:
            with open(results_filename, 'a') as f:
                f.write(f"Train Loss: {avg_epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}\n")
            print(f"Epoch {epoch+1} finished. Train Loss: {avg_epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            
    print(f"\nMetrics for {dataset_name} saved to: {results_filename}")
    print(f"Best test accuracy for {dataset_name}: {best_test_acc:.4f}")
    return history

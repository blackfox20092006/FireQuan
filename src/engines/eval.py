import jax
import jax.numpy as jnp
import optax
import numpy as np
from sklearn.metrics import f1_score, recall_score
from src.models.qnn import hybrid_model_forward

def evaluate_model(params, loader):
    total_loss = 0.0
    all_labels, all_predictions = [], []
    num_samples = 0
    
    @jax.jit
    def eval_step(params, images, labels):
        logits = hybrid_model_forward(params, images)
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

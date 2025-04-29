#%%
#Cell 1
# Cell 1: Imports and Setup
import os
import time
import math
import json
import random
import copy
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging  # Added for metric logging

from evaluation import load_model_and_metrics, evaluate_on_test_set
from test_visualisation import (
    create_loss_comparison_plot,
    create_improvement_heatmap,
    create_perplexity_comparison
)
from train_with_viz import test_configs_with_tracking, save_model_and_metrics
from new_gpt import ModelConfig, GPTLanguageModel, load_data, get_batch, estimate_loss
from gpt_downsizing import create_custom_config

# Detect device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Setup logging
logging.basicConfig(
    filename='experiment_metrics.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def set_seed(seed):
    """Set random seed for reproducibility across multiple runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Default seed for single runs
default_seed = 42
set_seed(default_seed)
#%%
def loss_function(x, minibatch):
    """
    Given parameter x in R^2, compute the average of
    min(13 * ||z||^2, ||z + [9,2]||^2)
    over all w in the minibatch, where z = x - w - 1.
    """
    y = 0.0
    for w in minibatch:
        z = x - w - 1  # z ∈ R^2
        val1 = 13.0 * (z[0]**2 + z[1]**2)
        val2 = (z[0] + 9.0)**2 + (z[1] + 2.0)**2
        y += min(val1, val2)
    return y / len(minibatch)
#%%
def gradient_function(x, minibatch):
    """
    Piecewise gradient of the loss_function with respect to x.
    If 13*||z||^2 < ||z + [9,2]||^2, use gradient of 13*||z||^2,
    else use gradient of ||z + [9,2]||^2.  Averaged over minibatch.
    """
    grad = np.zeros_like(x)
    for w in minibatch:
        z = x - w - 1
        val1 = 13.0 * (z[0]**2 + z[1]**2)
        val2 = (z[0] + 9.0)**2 + (z[1] + 2.0)**2

        if val1 < val2:
            # ∇(13||z||^2) = 26 z
            grad[0] += 26.0 * z[0]
            grad[1] += 26.0 * z[1]
        else:
            # ∇||z + [9,2]||^2 = 2*(z + [9,2])
            grad[0] += 2.0 * (z[0] + 9.0)
            grad[1] += 2.0 * (z[1] + 2.0)

    return grad / len(minibatch)
#%%
# Cell 4

def sgd_optimizer(
    data,
    x_init,
    loss_fn,
    grad_fn,
    method='constant',
    alpha=0.01,
    batch_size=8,
    max_epochs=100,
    beta1=0.9,         # for Heavy Ball and Adam
    beta2=0.999,       # for RMSProp and Adam
    epsilon=1e-8,      # for RMSProp and Adam
    polyak_f_star=0.0  # for Polyak steps
):
    """
    Perform mini-batch SGD on the given data using various step-size methods.

    :param data: array of shape (n, 2) representing n training points in R^2.
    :param x_init: initial parameter vector in R^2.
    :param loss_fn: f(x, minibatch) -> scalar.
    :param grad_fn: grad_f(x, minibatch) -> R^2 gradient.
    :param method: one of 'constant', 'polyak', 'rmsprop', 'heavy_ball', 'adam'.
    :param alpha: base step size (learning rate).
    :param batch_size: size of each mini-batch.
    :param max_epochs: number of passes through the data.
    :param beta1: momentum parameter (heavy ball, Adam).
    :param beta2: second-moment decay (RMSProp, Adam).
    :param epsilon: small constant to avoid division by zero.
    :param polyak_f_star: known (or estimated) minimal value of f.
    :return: (x, losses) final parameters and list of losses per epoch.
    """
    n = data.shape[0]
    x = x_init.copy()
    losses = []
    m = np.zeros_like(x)  # momentum / first moment
    v = np.zeros_like(x)  # second moment (for RMSProp/Adam)
    iteration = 0

    for epoch in range(max_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(n)
        data_shuffled = data[indices]

        # Mini-batch loop
        for start_idx in range(0, n, batch_size):
            minibatch = data_shuffled[start_idx:start_idx + batch_size]
            grad = grad_fn(x, minibatch)

            if method == 'constant':
                x -= alpha * grad

            elif method == 'polyak':
                current_loss = loss_fn(x, minibatch)
                denom = np.dot(grad, grad) + 1e-12
                alpha_k = (current_loss - polyak_f_star) / denom
                if alpha_k < alpha: 
                    print(f"[WARN] α_k={alpha_k:.2e}<α_min"); 
                    alpha_k = alpha
                x -= alpha_k * grad

            elif method == 'rmsprop':
                v = beta2 * v + (1 - beta2) * (grad * grad)
                x -= alpha * grad / (np.sqrt(v) + epsilon)

            elif method == 'heavy_ball':
                m = beta1 * m + alpha * grad
                x -= m

            elif method == 'adam':
                iteration += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                m_hat = m / (1 - beta1**iteration)
                v_hat = v / (1 - beta2**iteration)
                x -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        # Record full-dataset loss at end of epoch
        epoch_loss = loss_fn(x, data)
        losses.append(epoch_loss)

    return x, losses
#%%
# Cell 5: Mini-batch SGD with Polyak Step Size
def train_sgd(model, dataloader, loss_fn,
              num_epochs=20,
              step_method='constant',
              alpha=1e-2,
              f_star=0.0,
              alpha_max=1.0):  # Added upper bound
    """
    Mini-batch SGD training loop supporting:
      - constant step size (step_method='constant')
      - Polyak adaptive step size (step_method='polyak')
    
    Args:
        model: PyTorch model to train.
        dataloader: DataLoader for mini-batch data.
        loss_fn: Loss function (e.g., MSELoss, BCELoss).
        num_epochs: Number of epochs.
        step_method: 'constant' or 'polyak'.
        alpha: Learning rate (constant) or lower bound (Polyak).
        f_star: Assumed minimum loss (default 0.0 for Polyak).
        alpha_max: Upper bound on Polyak step size to prevent instability.
    
    Returns:
        history: List of per-epoch average losses.
        metrics: Dict with final loss and epochs to convergence.
    """
    model.to(device).train()
    history = []
    convergence_epoch = None
    threshold = 0.05  # Loss threshold for convergence
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            # Forward pass
            out = model(Xb)
            loss = loss_fn(out, yb)
            # Backward pass
            model.zero_grad()
            loss.backward()
            # Gather all gradients into one vector
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            
            if step_method == 'constant':
                lr = alpha
            else:  # 'polyak'
                loss_val = loss.item()
                # Compute gradient norm squared with small epsilon for stability
                # epsilon=1e-12 prevents division by zero when gradients are tiny
                denom = grads.dot(grads).item() + 1e-12
                # Polyak step size: (f_N(theta) - f_N*)/(||grad||^2 + epsilon)
                lr = (loss_val - f_star) / denom
                # Ensure step size is positive and bounded
                if lr < 0:
                    lr = alpha  # Fallback to lower bound if negative
                lr = max(alpha, min(lr, alpha_max))  # Bound between alpha and alpha_max
            
            # Manual parameter update
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad
            running_loss += loss.item() * Xb.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        history.append(epoch_loss)
        
        # Check for convergence
        if convergence_epoch is None and epoch_loss < threshold:
            convergence_epoch = epoch + 1
    
    # Log metrics
    metrics = {
        'final_loss': history[-1],
        'convergence_epoch': convergence_epoch or num_epochs
    }
    logging.info(f"train_sgd: step_method={step_method}, final_loss={history[-1]:.4f}, "
                 f"convergence_epoch={metrics['convergence_epoch']}")
    
    return history, metrics
#%%
# simple logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
#%%

# generate toy binary classification data
np.random.seed(0)
N = 1000
X = np.random.randn(N, 2)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1/(1+np.exp(-logits))
y = (probs > 0.5).astype(np.float32)

# wrap in DataLoader
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
ds = TensorDataset(X_t, y_t)
loader = DataLoader(ds, batch_size=32, shuffle=True)

#%%
# Cell 8: Binary Classification for 1(c)
# Generate toy binary classification data
np.random.seed(0)
torch.manual_seed(0)
N = 1000
X = np.random.randn(N, 2)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1/(1+np.exp(-logits))
y = (probs > 0.5).astype(np.float32)

# Wrap in DataLoader
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
ds = TensorDataset(X_t, y_t)
train_loader = DataLoader(ds, batch_size=32, shuffle=True)

bce = nn.BCELoss()

# Run multiple trials for robustness
num_trials = 3
hist_const_all = []
hist_poly_all = []
for trial in range(num_trials):
    set_seed(42 + trial)  # Different seed per trial
    
    # Constant-LR run
    model_const = LogisticRegression(2).to(device)
    hist_const, metrics_const = train_sgd(
        model_const, train_loader, bce,
        num_epochs=30,
        step_method='constant',
        alpha=0.1,
        alpha_max=1.0
    )
    
    # Polyak run
    model_poly = LogisticRegression(2).to(device)
    hist_poly, metrics_poly = train_sgd(
        model_poly, train_loader, bce,
        num_epochs=30,
        step_method='polyak',
        alpha=0.01,  # Lower bound
        f_star=0.0,
        alpha_max=1.0
    )
    
    hist_const_all.append(hist_const)
    hist_poly_all.append(hist_poly)
    
    # Log trial metrics
    logging.info(f"Trial {trial+1}: Constant LR - {metrics_const}")
    logging.info(f"Trial {trial+1}: Polyak LR - {metrics_poly}")

# Average results
hist_const = np.mean(hist_const_all, axis=0)
hist_poly = np.mean(hist_poly_all, axis=0)

# Plot training loss curves
plt.figure()
plt.plot(hist_const, label='Constant LR')
plt.plot(hist_poly, label='Polyak LR')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Binary Classification: Constant vs Polyak')
plt.legend()
plt.show()
#%%
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

#%%
# Cell 10: MLP Training for 1(c)
# Split data into train/test for validation
train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Run multiple trials
num_trials = 3
hist_const_all = []
hist_poly_all = []
val_loss_const_all = []
val_loss_poly_all = []

def compute_val_loss(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = loss_fn(out, yb)
            total_loss += loss.item() * Xb.size(0)
            total_samples += Xb.size(0)
    return total_loss / total_samples

for trial in range(num_trials):
    set_seed(42 + trial)
    
    # Constant-LR
    mlp_const = SimpleMLP(2).to(device)
    hist_const, metrics_const = train_sgd(
        mlp_const, train_loader, bce,
        num_epochs=30,
        step_method='constant',
        alpha=0.05,
        alpha_max=1.0
    )
    val_loss_const = compute_val_loss(mlp_const, test_loader, bce)
    
    # Polyak
    mlp_poly = SimpleMLP(2).to(device)
    hist_poly, metrics_poly = train_sgd(
        mlp_poly, train_loader, bce,
        num_epochs=30,
        step_method='polyak',
        alpha=0.05,
        f_star=0.0,
        alpha_max=1.0
    )
    val_loss_poly = compute_val_loss(mlp_poly, test_loader, bce)
    
    hist_const_all.append(hist_const)
    hist_poly_all.append(hist_poly)
    val_loss_const_all.append(val_loss_const)
    val_loss_poly_all.append(val_loss_poly)
    
    # Log trial metrics
    logging.info(f"MLP Trial {trial+1}: Constant - {metrics_const}, Val Loss: {val_loss_const:.4f}")
    logging.info(f"MLP Trial {trial+1}: Polyak - {metrics_poly}, Val Loss: {val_loss_poly:.4f}")

# Average results
hist_const = np.mean(hist_const_all, axis=0)
hist_poly = np.mean(hist_poly_all, axis=0)
val_loss_const = np.mean(val_loss_const_all)
val_loss_poly = np.mean(val_loss_poly_all)

# Plot training loss
plt.figure()
plt.plot(hist_const, label='MLP Constant LR')
plt.plot(hist_poly, label='MLP Polyak LR')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('MLP: Constant vs Polyak')
plt.legend()
plt.show()

# Log final metrics
logging.info(f"MLP Final: Constant LR - Train Loss: {hist_const[-1]:.4f}, Val Loss: {val_loss_const:.4f}")
logging.info(f"MLP Final: Polyak LR - Train Loss: {hist_poly[-1]:.4f}, Val Loss: {val_loss_poly:.4f}")
#%%

# simple linear model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(x)
#%%
# Cell 12: Linear Regression for 1(a) with Tests
'''
1a: Implementation and Testing of Polyak Step Size
'''
# Generate synthetic linear data y = 4x - 2 + noise
torch.manual_seed(0)
N = 200
X = torch.linspace(-5, 5, N).unsqueeze(1)
y = 4 * X - 2 + 0.5 * torch.randn_like(X)

# Wrap in DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=20, shuffle=True)

mse = nn.MSELoss(reduction='mean')

# Run multiple trials
num_trials = 3
hist_const_all = []
hist_poly_all = []
for trial in range(num_trials):
    set_seed(42 + trial)
    
    # Train with constant LR
    model_const = LinearModel().to(device)
    hist_const, metrics_const = train_sgd(
        model_const, loader, mse,
        num_epochs=50,
        step_method='constant',
        alpha=1e-2,
        alpha_max=1.0
    )
    
    # Train with Polyak step
    model_poly = LinearModel().to(device)
    hist_poly, metrics_poly = train_sgd(
        model_poly, loader, mse,
        num_epochs=50,
        step_method='polyak',
        alpha=1e-4,  # Lower bound
        f_star=0.0,
        alpha_max=1.0
    )
    
    hist_const_all.append(hist_const)
    hist_poly_all.append(hist_poly)
    
    # Log trial metrics
    logging.info(f"Linear Trial {trial+1}: Constant - {metrics_const}")
    logging.info(f"Linear Trial {trial+1}: Polyak - {metrics_poly}")

# Average results
hist_const = np.mean(hist_const_all, axis=0)
hist_poly = np.mean(hist_poly_all, axis=0)

# Plot losses
plt.figure()
plt.plot(hist_const, label='Constant LR')
plt.plot(hist_poly, label='Polyak Step')
plt.xlabel('Epoch')
plt.ylabel('Train MSE Loss')
plt.title('Linear Regression: Constant vs Polyak')
plt.legend()
plt.show()

# Print and log final loss values
print(f"Final Constant LR Loss: {hist_const[-1]:.4f}")
print(f"Final Polyak Loss: {hist_poly[-1]:.4f}")
logging.info(f"Linear Final: Constant LR Loss: {hist_const[-1]:.4f}")
logging.info(f"Linear Final: Polyak Loss: {hist_poly[-1]:.4f}")

# Test assertions
sigma = 0.5
assert hist_const[-1] < sigma**2 * 2, f"Constant LR MSE {hist_const[-1]} > {sigma**2 * 2}"
assert hist_poly[-1] < sigma**2 * 2, f"Polyak MSE {hist_poly[-1]} > {sigma**2 * 2}"
print("Tests passed: Final MSE within expected range")

# Edge-case test: Tiny gradients
X_edge = torch.linspace(-0.01, 0.01, N).unsqueeze(1)  # Small inputs
y_edge = 4 * X_edge - 2 + 0.01 * torch.randn_like(X_edge)  # Low noise
edge_loader = DataLoader(TensorDataset(X_edge, y_edge), batch_size=20, shuffle=True)

model_edge = LinearModel().to(device)
hist_edge, metrics_edge = train_sgd(
    model_edge, edge_loader, mse,
    num_epochs=50,
    step_method='polyak',
    alpha=1e-4,
    f_star=0.0,
    alpha_max=1.0
)
assert hist_edge[-1] < 0.01, f"Edge-case MSE {hist_edge[-1]} too high"
logging.info(f"Edge-case Test: Polyak MSE={hist_edge[-1]:.4f}, {metrics_edge}")
print(f"Edge-case Test Passed: Polyak MSE={hist_edge[-1]:.4f}")
#%%
# Cell 13: Linear Regression for 1(b)
'''
1b: Investigate Batch Sizes and Noise Levels
'''
torch.manual_seed(0)
np.random.seed(0)

batch_sizes = [5, 20, 100]
noise_levels = [0.1, 0.5, 1.0]
results = {}
num_trials = 3

# Hyperparameter tuning for constant LR
alpha_candidates = [1e-3, 1e-2, 5e-2]
best_alpha = 1e-2
best_mse = float('inf')
for alpha in alpha_candidates:
    mse_sum = 0
    for trial in range(num_trials):
        set_seed(42 + trial)
        X = torch.linspace(-5, 5, 200).unsqueeze(1)
        y = 4 * X - 2 + 0.5 * torch.randn_like(X)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=20, shuffle=True)
        model = LinearModel().to(device)
        hist, _ = train_sgd(model, loader, mse, num_epochs=50, step_method='constant', alpha=alpha)
        mse_sum += hist[-1]
    avg_mse = mse_sum / num_trials
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_alpha = alpha
print(f"Best Constant LR Alpha: {best_alpha}")

for noise_std in noise_levels:
    for batch_size in batch_sizes:
        hist_const_all = []
        hist_poly_all = []
        for trial in range(num_trials):
            set_seed(42 + trial)
            X = torch.linspace(-5, 5, 200).unsqueeze(1)
            y = 4 * X - 2 + noise_std * torch.randn_like(X)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Constant step size
            model_const = LinearModel().to(device)
            hist_const, metrics_const = train_sgd(
                model_const, loader, mse,
                num_epochs=50,
                step_method='constant',
                alpha=best_alpha,
                alpha_max=1.0
            )
            
            # Polyak step size
            model_poly = LinearModel().to(device)
            hist_poly, metrics_poly = train_sgd(
                model_poly, loader, mse,
                num_epochs=50,
                step_method='polyak',
                alpha=1e-4,
                f_star=0.0,
                alpha_max=1.0
            )
            
            hist_const_all.append(hist_const)
            hist_poly_all.append(hist_poly)
            
            # Log trial metrics
            logging.info(f"Noise={noise_std}, Batch={batch_size}, Trial {trial+1}: "
                         f"Constant - {metrics_const}")
            logging.info(f"Noise={noise_std}, Batch={batch_size}, Trial {trial+1}: "
                         f"Polyak - {metrics_poly}")
        
        # Average results
        results[f"noise_{noise_std}_batch_{batch_size}"] = {
            'constant': np.mean(hist_const_all, axis=0),
            'polyak': np.mean(hist_poly_all, axis=0),
            'final_const_mse': np.mean([h[-1] for h in hist_const_all]),
            'final_poly_mse': np.mean([h[-1] for h in hist_poly_all])
        }
        
        # Log final metrics
        logging.info(f"Noise={noise_std}, Batch={batch_size}: "
                     f"Final Constant MSE={results[f'noise_{noise_std}_batch_{batch_size}']['final_const_mse']:.4f}")
        logging.info(f"Noise={noise_std}, Batch={batch_size}: "
                     f"Final Polyak MSE={results[f'noise_{noise_std}_batch_{batch_size}']['final_poly_mse']:.4f}")

# Plot results
fig, axes = plt.subplots(len(noise_levels), len(batch_sizes), figsize=(18, 12), sharey=True)
for i, noise_std in enumerate(noise_levels):
    for j, batch_size in enumerate(batch_sizes):
        ax = axes[i, j]
        key = f"noise_{noise_std}_batch_{batch_size}"
        ax.plot(results[key]['constant'], label='Constant LR')
        ax.plot(results[key]['polyak'], label='Polyak LR')
        ax.set_title(f"Noise: {noise_std}, Batch: {batch_size}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
plt.tight_layout()
plt.show()
#%%
# Cell 14: Compare SGD with Constant vs Polyak Step Size for 1(c)
'''
1(c): Compare SGD with constant step size vs Polyak’s step size on week-6 binary classification task
'''


# Ensure train_sgd and SimpleMLP are defined (from previous cells)
# Assuming SimpleMLP is in Cell 9 and train_sgd in Cell 5

# Generate synthetic binary classification data (week-6 task)
torch.manual_seed(0)
np.random.seed(0)
N = 1000
X = np.random.randn(N, 2)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (probs > 0.5).astype(np.float32)

# Wrap in DataLoader with train/test split
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_t, y_t)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

bce = torch.nn.BCELoss()

# Hyperparameter tuning for constant LR
alpha_candidates = [0.01, 0.1, 0.5]
best_alpha = 0.1
best_loss = float('inf')
num_trials_tune = 3
for alpha in alpha_candidates:
    losses = []
    for trial in range(num_trials_tune):
        set_seed(42 + trial)
        model = SimpleMLP(2).to(device)
        hist, _ = train_sgd(model, train_loader, bce, num_epochs=50, step_method='constant', alpha=alpha, alpha_max=1.0)
        losses.append(hist[-1])
    avg_loss = np.mean(losses)
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_alpha = alpha
print(f"Best Constant LR Alpha: {best_alpha}")

# Run multiple trials for robustness
num_trials = 3
hist_const_all = []
hist_poly_all = []
val_loss_const_all = []
val_loss_poly_all = []
acc_const_all = []
acc_poly_all = []

def compute_val_loss(model, loader, loss_fn):
    """Compute validation loss for a model."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = loss_fn(out, yb)
            total_loss += loss.item() * Xb.size(0)
            total_samples += Xb.size(0)
    return total_loss / total_samples

def compute_accuracy(model, loader):
    """Compute accuracy for a model."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = (model(Xb) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

for trial in range(num_trials):
    set_seed(42 + trial)
    
    # Train with constant LR
    model_const = SimpleMLP(2).to(device)
    hist_const, metrics_const = train_sgd(
        model_const, train_loader, bce,
        num_epochs=50,
        step_method='constant',
        alpha=best_alpha,
        alpha_max=1.0
    )
    val_loss_const = compute_val_loss(model_const, test_loader, bce)
    acc_const = compute_accuracy(model_const, test_loader)
    
    # Train with Polyak step size
    model_poly = SimpleMLP(2).to(device)
    hist_poly, metrics_poly = train_sgd(
        model_poly, train_loader, bce,
        num_epochs=50,
        step_method='polyak',
        alpha=0.01,  # Lower bound
        f_star=0.0,
        alpha_max=1.0
    )
    val_loss_poly = compute_val_loss(model_poly, test_loader, bce)
    acc_poly = compute_accuracy(model_poly, test_loader)
    
    hist_const_all.append(hist_const)
    hist_poly_all.append(hist_poly)
    val_loss_const_all.append(val_loss_const)
    val_loss_poly_all.append(val_loss_poly)
    acc_const_all.append(acc_const)
    acc_poly_all.append(acc_poly)
    
    # Log trial metrics
    logging.info(f"1(c) Trial {trial+1}: Constant - Train Loss={hist_const[-1]:.4f}, "
                 f"Val Loss={val_loss_const:.4f}, Accuracy={acc_const:.4f}, {metrics_const}")
    logging.info(f"1(c) Trial {trial+1}: Polyak - Train Loss={hist_poly[-1]:.4f}, "
                 f"Val Loss={val_loss_poly:.4f}, Accuracy={acc_poly:.4f}, {metrics_poly}")

# Average results
hist_const = np.mean(hist_const_all, axis=0)
hist_poly = np.mean(hist_poly_all, axis=0)
val_loss_const = np.mean(val_loss_const_all)
val_loss_poly = np.mean(val_loss_poly_all)
acc_const = np.mean(acc_const_all)
acc_poly = np.mean(acc_poly_all)

# Plot training and validation loss curves
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_const, label='Constant LR')
plt.plot(hist_poly, label='Polyak LR')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('1(c) Training Loss: Constant vs Polyak')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([val_loss_const] * len(hist_const), label='Constant LR Val')
plt.plot([val_loss_poly] * len(hist_poly), label='Polyak LR Val')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('1(c) Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Print and log final metrics
print(f"Constant LR: Train Loss={hist_const[-1]:.4f}, Val Loss={val_loss_const:.4f}, Test Accuracy={acc_const:.4f}")
print(f"Polyak LR: Train Loss={hist_poly[-1]:.4f}, Val Loss={val_loss_poly:.4f}, Test Accuracy={acc_poly:.4f}")
logging.info(f"1(c) Final: Constant LR - Train Loss={hist_const[-1]:.4f}, "
             f"Val Loss={val_loss_const:.4f}, Test Accuracy={acc_const:.4f}")
logging.info(f"1(c) Final: Polyak LR - Train Loss={hist_poly[-1]:.4f}, "
             f"Val Loss={val_loss_poly:.4f}, Test Accuracy={acc_poly:.4f}")

# Test assertions
assert hist_const[-1] < 0.1, f"Constant LR train loss {hist_const[-1]} too high"
assert hist_poly[-1] < 0.1, f"Polyak LR train loss {hist_poly[-1]} too high"
assert acc_const > 0.95, f"Constant LR accuracy {acc_const} too low"
assert acc_poly > 0.95, f"Polyak LR accuracy {acc_poly} too low"
print("Tests passed: Loss and accuracy within expected range")
#%%
# Cell 15: Transformer Training for 1(d)
'''
1(d): Train week 9 GPT-style transformer with constant SGD, Polyak SGD, and AdamW.
Compare performance and analyze overfitting.
'''



# Prepare config and data
config = create_custom_config()
train_data, val_data, encode, decode, vocab_size = load_data(config)

# Training settings
num_iters = 2000
eval_interval = 100
polyak_f_star = 0.0
num_trials = 3
eps = 1e-12  # Numerical stability for Polyak

# Helper to do one full training run
def train_transformer(method, lr, betas=None):
    """
    Train transformer with specified method (constant, polyak, or adam).
    
    Args:
        method: 'constant', 'polyak', or 'adam'
        lr: Learning rate (constant/adam) or lower bound (polyak)
        betas: (beta1, beta2) for AdamW, if applicable
    
    Returns:
        train_losses, val_losses: Lists of train and validation losses
    """
    model = GPTLanguageModel(config, vocab_size).to(config.device)
    if method == 'constant':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif method == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas or (0.9, 0.999))
    
    train_losses, val_losses = [], []
    
    # Create progress bar
    pbar = tqdm(range(num_iters), desc=f"Training ({method})")
    
    for it in pbar:
        if it % eval_interval == 0:
            losses = estimate_loss(model, config, train_data, val_data)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            # Update progress bar with current losses
            pbar.set_postfix({
                'train_loss': f"{losses['train']:.4f}", 
                'val_loss': f"{losses['val']:.4f}"
            })
        
        xb, yb = get_batch(config, 'train', train_data, val_data)
        logits, loss = model(xb, yb)
        model.zero_grad()
        loss.backward()
        
        if method == 'polyak':
            # Compute Polyak step size: (f_N(theta) - f_N*)/(||grad||^2 + eps)
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            denom = grads.dot(grads).item() + eps
            current_loss = loss.item()
            alpha_k = max(min((current_loss - polyak_f_star) / denom, 1.0), lr)  # Bound alpha_k
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= alpha_k * p.grad
        else:
            optimizer.step()
    
    return train_losses, val_losses

# Hyperparameter tuning for AdamW
lr_candidates = {'constant': [5e-3, 1e-2, 2e-2], 'adam': [1e-4, 5e-4, 1e-3]}
beta2_candidates = [0.98, 0.999]
best_params = {'constant': 1e-2, 'adam': 5e-4, 'beta2': 0.98}
best_val_loss = float('inf')

print("Starting hyperparameter tuning...")
for lr_const in tqdm(lr_candidates['constant'], desc="Constant LR"):
    for lr_adam in tqdm(lr_candidates['adam'], desc="Adam LR", leave=False):
        for beta2 in tqdm(beta2_candidates, desc="Beta2", leave=False):
            val_losses = []
            for trial in range(num_trials):
                set_seed(42 + trial)
                _, val_loss = train_transformer('adam', lr_adam, betas=(0.9, beta2))
                val_losses.append(val_loss[-1])
            avg_val_loss = np.mean(val_losses)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_params = {'constant': lr_const, 'adam': lr_adam, 'beta2': beta2}
            
            # Print current results
            print(f"Params: const_lr={lr_const}, adam_lr={lr_adam}, beta2={beta2}, avg_val_loss={avg_val_loss:.4f}")

print(f"Best Params: {best_params}")

# Run all three methods
hist_const_all = []
hist_poly_all = []
hist_adam_all = []
val_const_all = []
val_poly_all = []
val_adam_all = []

print("Running final training with best parameters...")
for trial in tqdm(range(num_trials), desc="Trials"):
    set_seed(42 + trial)
    
    print(f"\nTrial {trial+1}/{num_trials}:")
    print("Training with Constant SGD...")
    hist_const, val_const = train_transformer('constant', lr=best_params['constant'])
    
    print("Training with Polyak SGD...")
    hist_poly, val_poly = train_transformer('polyak', lr=1e-4)
    
    print("Training with AdamW...")
    hist_adam, val_adam = train_transformer('adam', lr=best_params['adam'], 
                                           betas=(0.9, best_params['beta2']))
    
    hist_const_all.append(hist_const)
    hist_poly_all.append(hist_poly)
    hist_adam_all.append(hist_adam)
    val_const_all.append(val_const)
    val_poly_all.append(val_poly)
    val_adam_all.append(val_adam)
    
    # Log trial metrics
    logging.info(f"1(d) Trial {trial+1}: Constant - Train={hist_const[-1]:.4f}, Val={val_const[-1]:.4f}")
    logging.info(f"1(d) Trial {trial+1}: Polyak - Train={hist_poly[-1]:.4f}, Val={val_poly[-1]:.4f}")
    logging.info(f"1(d) Trial {trial+1}: Adam - Train={hist_adam[-1]:.4f}, Val={val_adam[-1]:.4f}")

# Average results
hist_const = np.mean(hist_const_all, axis=0)
hist_poly = np.mean(hist_poly_all, axis=0)
hist_adam = np.mean(hist_adam_all, axis=0)
val_const = np.mean(val_const_all, axis=0)
val_poly = np.mean(val_poly_all, axis=0)
val_adam = np.mean(val_adam_all, axis=0)

# Compute overfitting metrics
overfit_const = val_const[-1] - hist_const[-1]
overfit_poly = val_poly[-1] - hist_poly[-1]
overfit_adam = val_adam[-1] - hist_adam[-1]

# Plot comparison
steps = list(range(0, num_iters, eval_interval))
if len(steps) > len(hist_const):
    steps = steps[:len(hist_const)]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(steps, hist_const, label=f'SGD Constant (lr={best_params["constant"]})')
plt.plot(steps, hist_poly, label='SGD Polyak (lr_lb=1e-4)')
plt.plot(steps, hist_adam, label=f'AdamW (lr={best_params["adam"]})')
plt.xlabel('Iteration')
plt.ylabel('Train Loss')
plt.title('Transformer: Train Loss Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(steps, val_const, label='SGD Constant')
plt.plot(steps, val_poly, label='SGD Polyak')
plt.plot(steps, val_adam, label='AdamW')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.title('Transformer: Validation Loss Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# Print and log final metrics
print(f"Constant SGD: Train Loss={hist_const[-1]:.4f}, Val Loss={val_const[-1]:.4f}, Overfitting={overfit_const:.4f}")
print(f"Polyak SGD: Train Loss={hist_poly[-1]:.4f}, Val Loss={val_poly[-1]:.4f}, Overfitting={overfit_poly:.4f}")
print(f"AdamW: Train Loss={hist_adam[-1]:.4f}, Val Loss={val_adam[-1]:.4f}, Overfitting={overfit_adam:.4f}")
logging.info(f"1(d) Final: Constant - Train={hist_const[-1]:.4f}, Val={val_const[-1]:.4f}, Overfitting={overfit_const:.4f}")
logging.info(f"1(d) Final: Polyak - Train={hist_poly[-1]:.4f}, Val={val_poly[-1]:.4f}, Overfitting={overfit_poly:.4f}")
logging.info(f"1(d) Final: Adam - Train={hist_adam[-1]:.4f}, Val={val_adam[-1]:.4f}, Overfitting={overfit_adam:.4f}")

# Test assertions
assert hist_adam[-1] < 1.0, f"AdamW train loss {hist_adam[-1]} too high"
assert val_adam[-1] < 1.0, f"AdamW val loss {val_adam[-1]} too high"
print("Tests passed: AdamW losses within expected range")
#%%
# Cell 16: Polyak-Adam vs Adam for 1(e)
'''
1(e): Brief investigation of Polyak-Adam vs Adam on noisy linear regression.
'''



# Synthetic data (same as 1(b))
torch.manual_seed(0)
np.random.seed(0)
N = 200
X = torch.linspace(-5, 5, N).unsqueeze(1)
y = 4 * X - 2 + 0.5 * torch.randn_like(X)

# Split into train/test
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
train_ds, test_ds = random_split(TensorDataset(X, y), [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=20)

mse = nn.MSELoss()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Polyak-Adam implementation
def train_polyak_adam(model, dataloader, loss_fn, num_epochs=50, beta1=0.9, beta2=0.999, 
                      eps=1e-8, lr_lb=1e-4, f_star=0.0, alpha_max=1.0, device='cpu'):
    """
    Train model with Polyak-Adam (Polyak step size with Adam momentum).
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for training data
        loss_fn: Loss function (e.g., MSELoss)
        num_epochs: Number of epochs
        beta1, beta2: Adam momentum parameters
        eps: Adam numerical stability term
        lr_lb: Lower bound on Polyak step size
        f_star: Assumed minimum loss (0.0)
        alpha_max: Upper bound on Polyak step size
        device: Device for training
    
    Returns:
        history: List of per-epoch losses
        metrics: Dict with final loss and convergence epoch
    """
    model.to(device).train()
    m = [torch.zeros_like(p, device=device) for p in model.parameters()]
    v = [torch.zeros_like(p, device=device) for p in model.parameters()]
    step, history = 0, []
    convergence_epoch = None
    threshold = 0.5  # MSE threshold for convergence
    
    for epoch in range(num_epochs):
        running = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            step += 1
            pred = model(xb)
            loss = loss_fn(pred, yb)
            model.zero_grad()
            loss.backward()
            
            # Compute Polyak step size
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            gnorm2 = grads.dot(grads) + 1e-12
            alpha_k = max(min((loss.item() - f_star) / gnorm2.item(), alpha_max), lr_lb)
            
            # Adam update with Polyak step size
            with torch.no_grad():
                for i, p in enumerate(model.parameters()):
                    if p.grad is None: continue
                    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                    v[i] = beta2 * v[i] + (1 - beta2) * (p.grad * p.grad)
                    m_hat = m[i] / (1 - beta1 ** step)
                    v_hat = v[i] / (1 - beta2 ** step)
                    p -= alpha_k * m_hat / (v_hat.sqrt() + eps)
            
            running += loss.item() * xb.size(0)
        
        epoch_loss = running / len(dataloader.dataset)
        history.append(epoch_loss)
        if convergence_epoch is None and epoch_loss < threshold:
            convergence_epoch = epoch + 1
    
    metrics = {'final_loss': history[-1], 'convergence_epoch': convergence_epoch or num_epochs}
    return history, metrics

# Hyperparameter tuning for Adam
lr_candidates = [1e-3, 1e-2, 5e-2]
best_lr = 1e-2
best_mse = float('inf')
num_trials = 3
for lr in lr_candidates:
    mse_sum = 0
    for trial in range(num_trials):
        set_seed(42 + trial)
        model = LinearModel().to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        hist = []
        for epoch in range(50):
            running = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = mse(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
            hist.append(running / len(train_loader.dataset))
        mse_sum += hist[-1]
    if mse_sum / num_trials < best_mse:
        best_mse = mse_sum / num_trials
        best_lr = lr
print(f"Best Adam LR: {best_lr}")

# Run multiple trials
hist_adam_all = []
hist_padam_all = []
val_adam_all = []
val_padam_all = []

def compute_val_loss(model, loader, loss_fn):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = loss_fn(out, yb)
            total_loss += loss.item() * Xb.size(0)
            total_samples += Xb.size(0)
    return total_loss / total_samples

for trial in range(num_trials):
    set_seed(42 + trial)
    
    # Adam baseline
    model_adam = LinearModel().to(device)
    opt = optim.Adam(model_adam.parameters(), lr=best_lr)
    hist_adam = []
    for epoch in range(50):
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_adam(xb)
            loss = mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        hist_adam.append(running / len(train_loader.dataset))
    val_adam = compute_val_loss(model_adam, test_loader, mse)
    
    # Polyak-Adam
    model_padam = LinearModel().to(device)
    hist_padam, metrics_padam = train_polyak_adam(
        model_padam, train_loader, mse,
        num_epochs=50, lr_lb=1e-4, alpha_max=1.0, device=device
    )
    val_padam = compute_val_loss(model_padam, test_loader, mse)
    
    hist_adam_all.append(hist_adam)
    hist_padam_all.append(hist_padam)
    val_adam_all.append(val_adam)
    val_padam_all.append(val_padam)
    
    # Log trial metrics
    logging.info(f"1(e) Trial {trial+1}: Adam - Train={hist_adam[-1]:.4f}, Val={val_adam:.4f}")
    logging.info(f"1(e) Trial {trial+1}: Polyak-Adam - Train={hist_padam[-1]:.4f}, Val={val_padam:.4f}, {metrics_padam}")

# Average results
hist_adam = np.mean(hist_adam_all, axis=0)
hist_padam = np.mean(hist_padam_all, axis=0)
val_adam = np.mean(val_adam_all)
val_padam = np.mean(val_padam_all)

# Compute overfitting metrics
overfit_adam = val_adam - hist_adam[-1]
overfit_padam = val_padam - hist_padam[-1]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(hist_adam, label=f'Adam (lr={best_lr})')
plt.plot(hist_padam, label='Polyak-Adam (lr_lb=1e-4)')
plt.xlabel('Epoch')
plt.ylabel('Train MSE')
plt.title('1(e) Polyak-Adam vs Adam')
plt.legend()
plt.show()

# Print and log final metrics
print(f"Adam: Train MSE={hist_adam[-1]:.4f}, Val MSE={val_adam:.4f}, Overfitting={overfit_adam:.4f}")
print(f"Polyak-Adam: Train MSE={hist_padam[-1]:.4f}, Val MSE={val_padam:.4f}, Overfitting={overfit_padam:.4f}")
logging.info(f"1(e) Final: Adam - Train MSE={hist_adam[-1]:.4f}, Val MSE={val_adam:.4f}, Overfitting={overfit_adam:.4f}")
logging.info(f"1(e) Final: Polyak-Adam - Train MSE={hist_padam[-1]:.4f}, Val MSE={val_padam:.4f}, Overfitting={overfit_padam:.4f}")

# Test assertions
sigma = 0.5
assert hist_adam[-1] < sigma**2 * 2, f"Adam MSE {hist_adam[-1]} too high"
assert hist_padam[-1] < sigma**2 * 2, f"Polyak-Adam MSE {hist_padam[-1]} too high"
print("Tests passed: MSE within expected range")
#%%

#%%

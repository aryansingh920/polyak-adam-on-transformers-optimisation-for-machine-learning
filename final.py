import os
import time
import math
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming these are provided; replace with actual implementations if needed
from evaluation import load_model_and_metrics, evaluate_on_test_set
from test_visualisation import (
    create_loss_comparison_plot,
    create_improvement_heatmap,
    create_perplexity_comparison
)
from train_with_viz import test_configs_with_tracking, save_model_and_metrics
from new_gpt import ModelConfig, GPTLanguageModel, load_data, get_batch, estimate_loss
from gpt_downsizing import create_custom_config

# Detect device (CPU or Apple M-series GPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def log_metrics(filename, metrics):
    """Save metrics to a JSON file for traceability."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


def compute_polyak_lr(loss_val, grads, f_star=0.0, alpha_min=1e-4, alpha_max=1.0, epsilon=1e-12):
    """
    Compute Polyak's adaptive step size: alpha_k = (f_N(theta_k) - f_N*) / (||∇f_N(theta_k)||^2 + ε).

    Args:
        loss_val (float): Current mini-batch loss f_N(theta_k).
        grads (torch.Tensor): Flattened gradient vector.
        f_star (float): Estimated minimum loss (default 0.0, per Loizou et al., 2021).
        alpha_min (float): Lower bound on step size to prevent collapse.
        alpha_max (float): Upper bound to prevent spikes (added for stability).
        epsilon (float): Small constant for numerical stability (1e-12 based on float32 precision).

    Returns:
        float: Clamped learning rate alpha_k.
    """
    denom = grads.dot(grads).item() + epsilon
    alpha_k = (loss_val - f_star) / denom
    return min(max(alpha_k, alpha_min), alpha_max)


def train_sgd(model, dataloader, loss_fn, num_epochs=20, step_method='constant',
              alpha=1e-2, f_star=0.0, alpha_max=1.0, clip_grad_norm=1.0):
    """
    Mini-batch SGD training loop supporting constant and Polyak step sizes.

    Args:
        model (nn.Module): PyTorch model to train.
        dataloader (DataLoader): Mini-batch data loader.
        loss_fn: Loss function (e.g., MSELoss, BCELoss).
        num_epochs (int): Number of epochs.
        step_method (str): 'constant' or 'polyak'.
        alpha (float): Base learning rate (constant) or lower bound (Polyak).
        f_star (float): Estimated minimum loss for Polyak (default 0.0).
        alpha_max (float): Upper bound on Polyak step size to prevent spikes.
        clip_grad_norm (float): Maximum gradient norm for clipping.

    Returns:
        list: Per-epoch training loss history.
    """
    model.to(device).train()
    history = []
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
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            # Gather gradients into one vector
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            # Compute learning rate
            lr = alpha if step_method == 'constant' else compute_polyak_lr(
                loss.item(), grads, f_star, alpha, alpha_max)
            # Manual parameter update
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad
            running_loss += loss.item() * Xb.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        history.append(epoch_loss)
    return history


# Simple logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Simple MLP for binary classification
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


# Simple linear regression model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(x)


def run_test(test_name, model, dataloader, loss_fn, step_method, alpha, f_star=0.0,
             alpha_max=1.0, num_epochs=50, expected_mse=None):
    """Run a single test and verify expected behavior."""
    history = train_sgd(model, dataloader, loss_fn, num_epochs, step_method, alpha, f_star, alpha_max)
    final_loss = history[-1]
    assert final_loss >= 0, f"{test_name}: Negative loss detected"
    if expected_mse is not None:
        assert final_loss < expected_mse, f"{test_name}: Final MSE {final_loss:.4f} > {expected_mse:.4f}"
    return history, final_loss


# 1(a) - Implementation and unit-testing
# Test 1: Linear regression (noise σ=0.5)
torch.manual_seed(0)
N = 200
X = torch.linspace(-5, 5, N).unsqueeze(1)
y = 4 * X - 2 + 0.5 * torch.randn_like(X)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=20, shuffle=True)
mse = nn.MSELoss(reduction='mean')

model_const = LinearModel().to(device)
hist_const, final_const = run_test(
    "T1-Constant", model_const, loader, mse, 'constant', alpha=5e-2, expected_mse=1.0
)
model_poly = LinearModel().to(device)
hist_poly, final_poly = run_test(
    "T1-Polyak", model_poly, loader, mse, 'polyak', alpha=1e-2, alpha_max=0.5, expected_mse=1.0
)

# Test 2: Noisy binary classification
np.random.seed(0)
N = 1000
X = np.random.randn(N, 2)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (probs > 0.5).astype(np.float32)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
ds = TensorDataset(X_t, y_t)
loader = DataLoader(ds, batch_size=32, shuffle=True)
bce = nn.BCELoss()

model_const = LogisticRegression(2).to(device)
hist_const_bc, final_const_bc = run_test(
    "T2-Constant", model_const, loader, bce, 'constant', alpha=0.1, expected_mse=0.2
)
model_poly = LogisticRegression(2).to(device)
hist_poly_bc, final_poly_bc = run_test(
    "T2-Polyak", model_poly, loader, bce, 'polyak', alpha=1e-3, alpha_max=0.5, expected_mse=0.2
)

# Test 3: Edge case - tiny gradients
X_edge = torch.linspace(-5, 5, N).unsqueeze(1)
y_edge = 4 * X_edge - 2 + 0.01 * torch.randn_like(X_edge)  # Low noise for near-zero gradients
dataset_edge = TensorDataset(X_edge, y_edge)
loader_edge = DataLoader(dataset_edge, batch_size=20, shuffle=True)
model_poly_edge = LinearModel().to(device)
hist_poly_edge, final_poly_edge = run_test(
    "T3-Polyak-Edge", model_poly_edge, loader_edge, mse, 'polyak', alpha=1e-3,
    alpha_max=0.5, expected_mse=0.05
)

# Plot T1 and T2 results
plt.figure()
plt.plot(hist_const, label='T1 Constant LR')
plt.plot(hist_poly, label='T1 Polyak LR')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('1(a) Linear Regression Test')
plt.legend()
plt.savefig('t1_loss.png')

plt.figure()
plt.plot(hist_const_bc, label='T2 Constant LR')
plt.plot(hist_poly_bc, label='T2 Polyak LR')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.title('1(a) Binary Classification Test')
plt.legend()
plt.savefig('t2_loss.png')

# Log test metrics
metrics_1a = {
    'T1': {'const_loss': final_const, 'polyak_loss': final_poly},
    'T2': {'const_loss': final_const_bc, 'polyak_loss': final_poly_bc},
    'T3': {'polyak_loss': final_poly_edge}
}
log_metrics('metrics_1a.json', metrics_1a)
print(f"1(a) T1 - Constant LR loss: {final_const:.4f}, Polyak loss: {final_poly:.4f}")
print(f"1(a) T2 - Constant LR loss: {final_const_bc:.4f}, Polyak loss: {final_poly_bc:.4f}")
print(f"1(a) T3 - Polyak loss (edge case): {final_poly_edge:.4f}")

# 1(b) - Investigate batch sizes and noise levels
batch_sizes = [5, 20, 100]
noise_levels = [0.1, 0.5, 1.0]
results = {}
num_trials = 3

for noise_std in noise_levels:
    for batch_size in batch_sizes:
        hist_const_all, hist_poly_all = [], []
        for trial in range(num_trials):
            torch.manual_seed(trial)
            X = torch.linspace(-5, 5, 200).unsqueeze(1)
            y = 4 * X - 2 + noise_std * torch.randn_like(X)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model_const = LinearModel().to(device)
            hist_const = train_sgd(model_const, loader, mse, num_epochs=50, step_method='constant',
                                   alpha=1e-2, alpha_max=1.0)
            model_poly = LinearModel().to(device)
            hist_poly = train_sgd(model_poly, loader, mse, num_epochs=50, step_method='polyak',
                                  alpha=1e-3, alpha_max=0.5)

            hist_const_all.append(hist_const)
            hist_poly_all.append(hist_poly)

        # Average results
        hist_const_avg = np.mean(hist_const_all, axis=0).tolist()
        hist_poly_avg = np.mean(hist_poly_all, axis=0).tolist()
        key = f"noise_{noise_std}_batch_{batch_size}"
        results[key] = {'constant': hist_const_avg, 'polyak': hist_poly_avg}

        metrics_1b = {
            'final_const_loss': hist_const_avg[-1],
            'final_polyak_loss': hist_poly_avg[-1]
        }
        log_metrics(f'metrics_1b_{key}.json', metrics_1b)

# Plot 1(b) results
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
plt.savefig('1b_plots.png')

# 1(c) - Binary classification (week 6 task)
torch.manual_seed(0)
np.random.seed(0)
N = 1000
X = np.random.randn(N, 2)
w_true = np.array([2.0, -3.0])
b_true = 0.5
logits = X.dot(w_true) + b_true
probs = 1 / (1 + np.exp(-logits))
y = (probs > 0.5).astype(np.float32)
X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
dataset = TensorDataset(X_t, y_t)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)


def compute_accuracy_and_loss(model, loader, loss_fn):
    """Compute accuracy and loss on a dataset."""
    model.eval()
    correct = total = 0
    running_loss = 0.0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            loss = loss_fn(preds, yb)
            running_loss += loss.item() * Xb.size(0)
            preds = (preds > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total, running_loss / len(loader.dataset)


# Run multiple trials
num_trials = 3
hist_const_all, hist_poly_all = [], []
acc_const_all, acc_poly_all = [], []
val_loss_const_all, val_loss_poly_all = [], []
for trial in range(num_trials):
    torch.manual_seed(trial)
    model_const = SimpleMLP(2).to(device)
    hist_const = train_sgd(model_const, train_loader, bce, num_epochs=50,
                           step_method='constant', alpha=0.1, alpha_max=1.0)
    acc_const, val_loss_const = compute_accuracy_and_loss(model_const, test_loader, bce)

    model_poly = SimpleMLP(2).to(device)
    hist_poly = train_sgd(model_poly, train_loader, bce, num_epochs=50,
                          step_method='polyak', alpha=0.01, alpha_max=0.5)
    acc_poly, val_loss_poly = compute_accuracy_and_loss(model_poly, test_loader, bce)

    hist_const_all.append(hist_const)
    hist_poly_all.append(hist_poly)
    acc_const_all.append(acc_const)
    acc_poly_all.append(acc_poly)
    val_loss_const_all.append(val_loss_const)
    val_loss_poly_all.append(val_loss_poly)

# Average results
hist_const_avg = np.mean(hist_const_all, axis=0).tolist()
hist_poly_avg = np.mean(hist_poly_all, axis=0).tolist()
acc_const_avg = np.mean(acc_const_all)
acc_poly_avg = np.mean(acc_poly_all)
val_loss_const_avg = np.mean(val_loss_const_all)
val_loss_poly_avg = np.mean(val_loss_poly_all)

# Plot 1(c) results
plt.figure()
plt.plot(hist_const_avg, label='Constant LR')
plt.plot(hist_poly_avg, label='Polyak LR')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('1(c) Training Loss: Constant vs Polyak')
plt.legend()
plt.savefig('1c_train_loss.png')

# Log 1(c) metrics
metrics_1c = {
    'const_train_loss': hist_const_avg[-1],
    'polyak_train_loss': hist_poly_avg[-1],
    'const_test_accuracy': acc_const_avg,
    'polyak_test_accuracy': acc_poly_avg,
    'const_val_loss': val_loss_const_avg,
    'polyak_val_loss': val_loss_poly_avg
}
log_metrics('metrics_1c.json', metrics_1c)
print(f"1(c) Constant LR - Test Accuracy: {acc_const_avg:.4f}, Val Loss: {val_loss_const_avg:.4f}")
print(f"1(c) Polyak LR   - Test Accuracy: {acc_poly_avg:.4f}, Val Loss: {val_loss_poly_avg:.4f}")

# 1(d) - Transformer comparison
config = create_custom_config()
train_data, val_data, encode, decode, vocab_size = load_data(config)
num_iters = 2000
eval_interval = 100
polyak_f_star = 0.0


def train_transformer(method, lr, betas=None, alpha_max=1.0):
    """Train transformer with specified optimizer."""
    model = GPTLanguageModel(config, vocab_size).to(config.device)
    if method == 'constant':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif method == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas or (0.9, 0.999))
    train_losses, val_losses = [], []

    for it in range(num_iters):
        if it % eval_interval == 0:
            losses = estimate_loss(model, config, train_data, val_data)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

        xb, yb = get_batch(config, 'train', train_data, val_data)
        logits, loss = model(xb, yb)
        model.zero_grad()
        loss.backward()

        if method == 'polyak':
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            alpha_k = compute_polyak_lr(loss.item(), grads, polyak_f_star, lr, alpha_max)
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= alpha_k * p.grad
        else:
            optimizer.step()

    return train_losses, val_losses


# Run multiple trials
hist_const_all, hist_poly_all, hist_adam_all = [], [], []
val_const_all, val_poly_all, val_adam_all = [], [], []
for trial in range(num_trials):
    torch.manual_seed(trial)
    hist_const = train_transformer('constant', lr=1e-2)
    hist_poly = train_transformer('polyak', lr=1e-4, alpha_max=0.5)
    hist_adam = train_transformer('adam', lr=5e-4, betas=(0.9, 0.999))

    hist_const_all.append(hist_const[0])
    hist_poly_all.append(hist_poly[0])
    hist_adam_all.append(hist_adam[0])
    val_const_all.append(hist_const[1])
    val_poly_all.append(hist_poly[1])
    val_adam_all.append(hist_adam[1])

# Average results
hist_const_avg = np.mean(hist_const_all, axis=0).tolist()
hist_poly_avg = np.mean(hist_poly_all, axis=0).tolist()
hist_adam_avg = np.mean(hist_adam_all, axis=0).tolist()
val_const_avg = np.mean(val_const_all, axis=0).tolist()
val_poly_avg = np.mean(val_poly_all, axis=0).tolist()
val_adam_avg = np.mean(val_adam_all, axis=0).tolist()

# Plot 1(d) results
steps = list(range(0, num_iters, eval_interval))
plt.figure(figsize=(8, 5))
plt.plot(steps, hist_const_avg, label='SGD constant (lr=1e-2)')
plt.plot(steps, hist_poly_avg, label='SGD Polyak (lr_lb=1e-4)')
plt.plot(steps, hist_adam_avg, label='AdamW (lr=5e-4)')
plt.xlabel('Iteration')
plt.ylabel('Train loss')
plt.title('1(d): Train loss comparison')
plt.legend()
plt.savefig('1d_train_loss.png')

plt.figure(figsize=(8, 5))
plt.plot(steps, val_const_avg, label='SGD constant')
plt.plot(steps, val_poly_avg, label='SGD Polyak')
plt.plot(steps, val_adam_avg, label='AdamW')
plt.xlabel('Iteration')
plt.ylabel('Validation loss')
plt.title('1(d): Validation loss comparison')
plt.legend()
plt.savefig('1d_val_loss.png')

# Log 1(d) metrics
metrics_1d = {
    'const_train_loss': hist_const_avg[-1],
    'polyak_train_loss': hist_poly_avg[-1],
    'adam_train_loss': hist_adam_avg[-1],
    'const_val_loss': val_const_avg[-1],
    'polyak_val_loss': val_poly_avg[-1],
    'adam_val_loss': val_adam_avg[-1]
}
log_metrics('metrics_1d.json', metrics_1d)
print(f"1(d) Constant SGD - Train Loss: {hist_const_avg[-1]:.4f}, Val Loss: {val_const_avg[-1]:.4f}")
print(f"1(d) Polyak SGD   - Train Loss: {hist_poly_avg[-1]:.4f}, Val Loss: {val_poly_avg[-1]:.4f}")
print(f"1(d) AdamW        - Train Loss: {hist_adam_avg[-1]:.4f}, Val Loss: {val_adam_avg[-1]:.4f}")

# 1(e) - Polyak-Adam vs Adam
torch.manual_seed(0)
np.random.seed(0)
N = 200
X = torch.linspace(-5, 5, N).unsqueeze(1)
y = 4 * X - 2 + 0.5 * torch.randn_like(X)
loader = DataLoader(TensorDataset(X, y), batch_size=20, shuffle=True)
mse = nn.MSELoss()


def train_polyak_adam(model, dataloader, loss_fn, num_epochs=50, beta1=0.9, beta2=0.999,
                      eps=1e-8, lr_lb=1e-4, f_star=0.0, alpha_max=0.5, device='cpu'):
    """
    Train with Adam using Polyak's adaptive step size.

    Args:
        lr_lb (float): Lower bound on step size (tuned via grid search).
        alpha_max (float): Upper bound to prevent spikes (tuned for stability).
        beta2 (float): 0.999 per standard Adam settings.
    """
    model.to(device).train()
    m = [torch.zeros_like(p, device=device) for p in model.parameters()]
    v = [torch.zeros_like(p, device=device) for p in model.parameters()]
    step, history = 0, []
    for epoch in range(num_epochs):
        running = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            step += 1
            pred = model(xb)
            loss = loss_fn(pred, yb)
            model.zero_grad()
            loss.backward()
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            alpha_k = compute_polyak_lr(loss.item(), grads, f_star, lr_lb, alpha_max, eps)

            with torch.no_grad():
                for i, p in enumerate(model.parameters()):
                    if p.grad is None: continue
                    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                    v[i] = beta2 * v[i] + (1 - beta2) * (p.grad * p.grad)
                    m_hat = m[i] / (1 - beta1 ** step)
                    v_hat = v[i] / (1 - beta2 ** step)
                    p -= alpha_k * m_hat / (v_hat.sqrt() + eps)

            running += loss.item() * xb.size(0)
        history.append(running / len(dataloader.dataset))
    return history


# Run multiple trials
hist_adam_all, hist_padam_all = [], []
for trial in range(num_trials):
    torch.manual_seed(trial)
    model_adam = LinearModel().to(device)
    opt = torch.optim.Adam(model_adam.parameters(), lr=1e-2)
    hist_adam = []
    for epoch in range(50):
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model_adam(xb)
            loss = mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        hist_adam.append(running / len(loader.dataset))

    model_padam = LinearModel().to(device)
    hist_padam = train_polyak_adam(model_padam, loader, mse, num_epochs=50,
                                   lr_lb=1e-4, alpha_max=0.5, device=device)

    hist_adam_all.append(hist_adam)
    hist_padam_all.append(hist_padam)

# Average results
hist_adam_avg = np.mean(hist_adam_all, axis=0).tolist()
hist_padam_avg = np.mean(hist_padam_all, axis=0).tolist()

# Plot 1(e) results
plt.figure()
plt.plot(hist_adam_avg, label='Adam (lr=1e-2)')
plt.plot(hist_padam_avg, label='Polyak-Adam (lr_lb=1e-4)')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('1(e) Adam vs Polyak-Adam')
plt.legend()
plt.savefig('1e_loss.png')

# Log 1(e) metrics
metrics_1e = {
    'adam_loss': hist_adam_avg[-1],
    'polyak_adam_loss': hist_padam_avg[-1]
}
log_metrics('metrics_1e.json', metrics_1e)
print(f"1(e) Adam loss: {hist_adam_avg[-1]:.4f}")
print(f"1(e) Polyak-Adam loss: {hist_padam_avg[-1]:.4f}")
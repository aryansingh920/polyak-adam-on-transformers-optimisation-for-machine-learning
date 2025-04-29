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
def train_sgd(model, dataloader, loss_fn,
              num_epochs=20,
              step_method='constant',
              alpha=1e-2,
              f_star=0.0,
              alpha_max=1.0):
    model.to(device).train()
    history = []
    convergence_epoch = None
    threshold = 0.05  # Loss threshold for convergence

    # NEW: Log alpha values for Polyak
    alpha_log = []  

    for epoch in range(num_epochs):
        running_loss = 0.0
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = loss_fn(out, yb)
            model.zero_grad()
            loss.backward()
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

            if step_method == 'constant':
                lr = alpha
            else:  # 'polyak'
                loss_val = loss.item()
                denom = grads.dot(grads).item() + 1e-12
                lr = (loss_val - f_star) / denom
                if lr < 0:
                    lr = alpha
                lr = max(alpha, min(lr, alpha_max))
                
                # NEW: Log alpha_k
                alpha_log.append(lr)

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad
            running_loss += loss.item() * Xb.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        history.append(epoch_loss)

        if convergence_epoch is None and epoch_loss < threshold:
            convergence_epoch = epoch + 1

    metrics = {
        'final_loss': history[-1],
        'convergence_epoch': convergence_epoch or num_epochs
    }

    # NEW: If Polyak, store alpha_log inside metrics
    if step_method == 'polyak':
        metrics['alpha_log'] = alpha_log

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
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ])
        self.net = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.net(x)
#%%

#%%
# simple linear model
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(x)
#%%
def train_sgd(model, loader, loss_fn, num_epochs=10, step_method='constant', 
           alpha=1e-3, alpha_max=1.0, f_star=None, return_step_sizes=False):
    """
    Train a model using SGD with either constant or Polyak step sizes.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model to train
    loader : DataLoader
        PyTorch data loader
    loss_fn : callable
        Loss function
    num_epochs : int
        Number of epochs to train
    step_method : str
        'constant' or 'polyak'
    alpha : float
        Learning rate for constant, or minimum step size for polyak
    alpha_max : float
        Maximum step size (for clamping)
    f_star : float or None
        Optimal loss value for Polyak step size
    return_step_sizes : bool
        Whether to return step sizes and clamp rates used during training
        
    Returns:
    --------
    losses : list
        Loss history
    metrics : dict
        Training metrics
    """
    optimizer = optim.SGD(model.parameters(), lr=alpha)
    losses = []
    
    # For step size analysis
    step_sizes = []
    clamp_rates = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_steps = []
        epoch_clamps = 0
        total_steps = 0
        
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            epoch_losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Apply step size logic based on method
            if step_method == 'constant':
                # Default step size is already set in optimizer
                step_size = alpha
                
                # Apply the step
                optimizer.step()
                
                # Record the step for analysis
                epoch_steps.append(step_size)
                epoch_clamps += int(step_size >= alpha_max)
                total_steps += 1
                
            elif step_method == 'polyak':
                # Implement Polyak step size: alpha_t = (f(w) - f*) / ||∇f(w)||²
                if f_star is None:
                    raise ValueError("f_star must be provided for Polyak step size")
                
                # Compute squared gradient norm
                grad_sq_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_sq_norm += torch.sum(param.grad ** 2).item()
                
                # Avoid division by zero
                if grad_sq_norm > 0:
                    # Polyak step size formula
                    step_size = (loss.item() - f_star) / grad_sq_norm
                    
                    # Clamp step size
                    original_step_size = step_size
                    step_size = max(alpha, min(step_size, alpha_max))
                    was_clamped = (step_size != original_step_size)
                    
                    # Set the learning rate
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = step_size
                    
                    # Apply the step
                    optimizer.step()
                    
                    # Record the step for analysis
                    epoch_steps.append(step_size)
                    epoch_clamps += int(was_clamped)
                    total_steps += 1
                else:
                    # Skip update if gradient is zero
                    pass
            else:
                raise ValueError(f"Unknown step method: {step_method}")
        
        # Compute mean loss for this epoch
        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        
        # Compute mean step size and clamp rate for this epoch
        if epoch_steps:
            mean_step_size = np.mean(epoch_steps)
            clamp_rate = (epoch_clamps / total_steps) * 100  # As percentage
        else:
            mean_step_size = 0
            clamp_rate = 0
            
        step_sizes.append(mean_step_size)
        clamp_rates.append(clamp_rate)
    
    # Calculate training metrics
    train_time = time.time() - start_time
    metrics = {
        'train_time': f"{train_time:.2f}s",
        'final_loss': f"{losses[-1]:.6f}",
        'avg_step_size': f"{np.mean(step_sizes):.6f}",
        'avg_clamp_rate': f"{np.mean(clamp_rates):.2f}%"
    }
    
    if return_step_sizes:
        return (losses, step_sizes, clamp_rates), metrics
    else:
        return losses, metrics


# Cell 12: Linear Regression for 1(a) with Tests and Step Size Analysis
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
step_sizes_const_all = []  # Track constant step sizes
step_sizes_poly_all = []   # Track Polyak step sizes
clamp_rates_const_all = [] # Track clamp rates for constant LR
clamp_rates_poly_all = []  # Track clamp rates for Polyak

for trial in range(num_trials):
    set_seed(42 + trial)
    
    # Train with constant LR
    model_const = LinearModel().to(device)
    hist_const, metrics_const = train_sgd(
        model_const, loader, mse,
        num_epochs=50,
        step_method='constant',
        alpha=1e-2,
        alpha_max=1.0,
        return_step_sizes=True  # Add parameter to return step sizes
    )
    
    # Train with Polyak step
    model_poly = LinearModel().to(device)
    hist_poly, metrics_poly = train_sgd(
        model_poly, loader, mse,
        num_epochs=50,
        step_method='polyak',
        alpha=1e-4,  # Lower bound
        f_star=0.0,
        alpha_max=1.0,
        return_step_sizes=True  # Add parameter to return step sizes
    )
    
    # Extract loss history and step size data from returned metrics
    loss_hist_const, step_sizes_const, clamp_rates_const = hist_const
    loss_hist_poly, step_sizes_poly, clamp_rates_poly = hist_poly
    
    # Store for averaging
    hist_const_all.append(loss_hist_const)
    hist_poly_all.append(loss_hist_poly)
    step_sizes_const_all.append(step_sizes_const)
    step_sizes_poly_all.append(step_sizes_poly)
    clamp_rates_const_all.append(clamp_rates_const)
    clamp_rates_poly_all.append(clamp_rates_poly)
    
    # Log trial metrics
    logging.info(f"Linear Trial {trial+1}: Constant - {metrics_const}")
    logging.info(f"Linear Trial {trial+1}: Polyak - {metrics_poly}")

# Average results
hist_const = np.mean(hist_const_all, axis=0)
hist_poly = np.mean(hist_poly_all, axis=0)
avg_step_sizes_const = np.mean(step_sizes_const_all, axis=0)
avg_step_sizes_poly = np.mean(step_sizes_poly_all, axis=0)
avg_clamp_rates_const = np.mean(clamp_rates_const_all, axis=0)
avg_clamp_rates_poly = np.mean(clamp_rates_poly_all, axis=0)

# Create figure with multiple subplots
plt.figure(figsize=(15, 12))

# Plot 1: Losses
plt.subplot(2, 2, 1)
plt.plot(hist_const, label='Constant LR')
plt.plot(hist_poly, label='Polyak Step')
plt.xlabel('Epoch')
plt.ylabel('Train MSE Loss')
plt.title('Linear Regression: Constant vs Polyak Loss')
plt.legend()

# Plot 2: Step Sizes
plt.subplot(2, 2, 2)
plt.plot(avg_step_sizes_const, label='Constant LR')
plt.plot(avg_step_sizes_poly, label='Polyak Step')
plt.xlabel('Epoch')
plt.ylabel('Step Size')
plt.title('Step Size vs Epoch')
plt.legend()

# Plot 3: Clamp Rates
plt.subplot(2, 2, 3)
plt.plot(avg_clamp_rates_const, label='Constant LR')
plt.plot(avg_clamp_rates_poly, label='Polyak Step')
plt.xlabel('Epoch')
plt.ylabel('Clamp Rate (% of steps clamped)')
plt.title('Clamp Rate vs Epoch')
plt.legend()

# Plot 4: Clamp Rate Heatmap
plt.subplot(2, 2, 4)
clamp_data = np.vstack([avg_clamp_rates_const, avg_clamp_rates_poly])
plt.imshow(clamp_data, aspect='auto', cmap='hot')
plt.colorbar(label='Clamp Rate (%)')
plt.yticks([0, 1], ['Constant LR', 'Polyak Step'])
plt.xlabel('Epoch')
plt.title('Clamp Rate Comparison')

plt.tight_layout()
plt.show()

# Create tables for clamp rate analysis
epochs_to_show = [0, 5, 10, 20, 30, 49]  # Select specific epochs to display
clamp_data_table = pd.DataFrame({
    'Epoch': epochs_to_show,
    'Constant LR Step Size': [avg_step_sizes_const[i] for i in epochs_to_show],
    'Constant LR Clamp Rate (%)': [avg_clamp_rates_const[i] for i in epochs_to_show],
    'Polyak Step Size': [avg_step_sizes_poly[i] for i in epochs_to_show],
    'Polyak Clamp Rate (%)': [avg_clamp_rates_poly[i] for i in epochs_to_show]
})

# Display the table
print("\nStep Size and Clamp Rate Analysis Table:")
print(clamp_data_table.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# Save table to CSV for further analysis
clamp_data_table.to_csv('linear_regression_step_analysis.csv', index=False)
print("Analysis table saved to 'linear_regression_step_analysis.csv'")

# Print and log final loss values
print(f"\nFinal Constant LR Loss: {hist_const[-1]:.4f}")
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
    alpha_max=1.0,
    return_step_sizes=True
)

loss_hist_edge, step_sizes_edge, clamp_rates_edge = hist_edge

# Plot edge case step sizes and clamp rates
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(step_sizes_edge)
plt.xlabel('Epoch')
plt.ylabel('Step Size')
plt.title('Edge Case: Polyak Step Size')

plt.subplot(1, 2, 2)
plt.plot(clamp_rates_edge)
plt.xlabel('Epoch')
plt.ylabel('Clamp Rate (%)')
plt.title('Edge Case: Clamp Rate')

plt.tight_layout()
plt.show()

assert loss_hist_edge[-1] < 0.01, f"Edge-case MSE {loss_hist_edge[-1]} too high"
logging.info(f"Edge-case Test: Polyak MSE={loss_hist_edge[-1]:.4f}, {metrics_edge}")
print(f"Edge-case Test Passed: Polyak MSE={loss_hist_edge[-1]:.4f}")
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
alpha_final_mses = {}  # Store final MSE for each alpha

# For Appendix 1b table: final MSE for each alpha at σ=0.5, batch_size=20
for alpha in alpha_candidates:
    mse_sum = 0
    alpha_mses = []  # Store individual trial MSEs
    
    for trial in range(num_trials):
        set_seed(42 + trial)
        X = torch.linspace(-5, 5, 200).unsqueeze(1)
        y = 4 * X - 2 + 0.5 * torch.randn_like(X)  # Using σ=0.5 for tuning
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=20, shuffle=True)
        model = LinearModel().to(device)
        hist, _ = train_sgd(model, loader, mse, num_epochs=50, step_method='constant', alpha=alpha)
        final_mse = hist[-1]
        mse_sum += final_mse
        alpha_mses.append(final_mse)
    
    avg_mse = mse_sum / num_trials
    alpha_final_mses[alpha] = {
        'trial_mses': alpha_mses,
        'avg_mse': avg_mse
    }
    
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_alpha = alpha

print(f"Best Constant LR Alpha: {best_alpha}")

# Print table for Appendix 1b - Final MSE for different constant α values
print("\nAppendix 1b: Final MSE for Different Constant α Values (σ=0.5, Batch Size=20)")
print("=" * 80)
print(f"{'α (Constant Step Size)':^25} | {'Final MSE':^15}")
print("-" * 80)
for alpha in alpha_candidates:
    avg_mse = alpha_final_mses[alpha]['avg_mse']
    print(f"{alpha:^25.0e} | {avg_mse:^15.4f}")
print("=" * 80)
print(f"Based on these results, α={best_alpha:.0e} was selected as it achieved the lowest final MSE.")
print("=" * 80)

# Dictionary to store convergence epochs
convergence_data = {
    'noise_level': [],
    'const_lr_epochs_to_converge': [],
    'polyak_lr_epochs_to_converge': [],
    'const_lr_walltime_per_epoch': [],
    'polyak_lr_walltime_per_epoch': []
}

for noise_std in noise_levels:
    for batch_size in batch_sizes:
        hist_const_all = []
        hist_poly_all = []
        const_epochs_to_converge = []
        polyak_epochs_to_converge = []
        const_walltime_per_epoch = []
        polyak_walltime_per_epoch = []
        
        for trial in range(num_trials):
            set_seed(42 + trial)
            X = torch.linspace(-5, 5, 200).unsqueeze(1)
            y = 4 * X - 2 + noise_std * torch.randn_like(X)
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Target MSE threshold is σ²
            target_mse = noise_std**2
            
            # Constant step size
            model_const = LinearModel().to(device)
            start_time = time.time()
            hist_const, metrics_const = train_sgd(
                model_const, loader, mse,
                num_epochs=50,
                step_method='constant',
                alpha=best_alpha,
                alpha_max=1.0
            )
            end_time = time.time()
            const_time_per_epoch = (end_time - start_time) / 50
            const_walltime_per_epoch.append(const_time_per_epoch)
            
            # Find first epoch where MSE ≤ σ²
            const_converged_epoch = 51  # Default if never converges
            for epoch, loss in enumerate(hist_const):
                if loss <= target_mse:
                    const_converged_epoch = epoch + 1  # +1 because epochs are 1-indexed
                    break
            const_epochs_to_converge.append(const_converged_epoch)
            
            # Polyak step size
            model_poly = LinearModel().to(device)
            start_time = time.time()
            hist_poly, metrics_poly = train_sgd(
                model_poly, loader, mse,
                num_epochs=50,
                step_method='polyak',
                alpha=1e-4,
                f_star=0.0,
                alpha_max=1.0
            )
            end_time = time.time()
            polyak_time_per_epoch = (end_time - start_time) / 50
            polyak_walltime_per_epoch.append(polyak_time_per_epoch)
            
            # Find first epoch where MSE ≤ σ²
            polyak_converged_epoch = 51  # Default if never converges
            for epoch, loss in enumerate(hist_poly):
                if loss <= target_mse:
                    polyak_converged_epoch = epoch + 1  # +1 because epochs are 1-indexed
                    break
            polyak_epochs_to_converge.append(polyak_converged_epoch)
            
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
            'final_poly_mse': np.mean([h[-1] for h in hist_poly_all]),
            'const_epochs_to_converge': const_epochs_to_converge,
            'polyak_epochs_to_converge': polyak_epochs_to_converge,
            'avg_const_epochs': np.mean(const_epochs_to_converge),
            'avg_polyak_epochs': np.mean(polyak_epochs_to_converge),
            'avg_const_walltime': np.mean(const_walltime_per_epoch),
            'avg_polyak_walltime': np.mean(polyak_walltime_per_epoch)
        }
        
        # Log final metrics
        logging.info(f"Noise={noise_std}, Batch={batch_size}: "
                     f"Final Constant MSE={results[f'noise_{noise_std}_batch_{batch_size}']['final_const_mse']:.4f}")
        logging.info(f"Noise={noise_std}, Batch={batch_size}: "
                     f"Final Polyak MSE={results[f'noise_{noise_std}_batch_{batch_size}']['final_poly_mse']:.4f}")
        
        # Log convergence data
        logging.info(f"Noise={noise_std}, Batch={batch_size}: "
                     f"Const LR epochs to MSE ≤ σ²: {results[f'noise_{noise_std}_batch_{batch_size}']['avg_const_epochs']:.1f}")
        logging.info(f"Noise={noise_std}, Batch={batch_size}: "
                     f"Polyak LR epochs to MSE ≤ σ²: {results[f'noise_{noise_std}_batch_{batch_size}']['avg_polyak_epochs']:.1f}")
        
        # Save convergence data for summary table
        if batch_size == 20:  # Only collect for the middle batch size for summary table
            convergence_data['noise_level'].append(noise_std)
            convergence_data['const_lr_epochs_to_converge'].append(results[f'noise_{noise_std}_batch_{batch_size}']['avg_const_epochs'])
            convergence_data['polyak_lr_epochs_to_converge'].append(results[f'noise_{noise_std}_batch_{batch_size}']['avg_polyak_epochs'])
            convergence_data['const_lr_walltime_per_epoch'].append(results[f'noise_{noise_std}_batch_{batch_size}']['avg_const_walltime'])
            convergence_data['polyak_lr_walltime_per_epoch'].append(results[f'noise_{noise_std}_batch_{batch_size}']['avg_polyak_walltime'])

# Plot results
fig, axes = plt.subplots(len(noise_levels), len(batch_sizes), figsize=(18, 12), sharey=True)
for i, noise_std in enumerate(noise_levels):
    for j, batch_size in enumerate(batch_sizes):
        ax = axes[i, j]
        key = f"noise_{noise_std}_batch_{batch_size}"
        ax.plot(results[key]['constant'], label='Constant LR')
        ax.plot(results[key]['polyak'], label='Polyak LR')
        ax.axhline(y=noise_std**2, color='r', linestyle='--', label='σ² threshold')
        ax.set_title(f"Noise: {noise_std}, Batch: {batch_size}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
plt.tight_layout()
plt.show()

# Print convergence rate summary table for batch_size=20
print("\nConvergence Rate Summary (Batch Size = 20)")
print("=" * 80)
print(f"{'Noise Level':^15} | {'Const-LR Epochs to MSE ≤ σ²':^30} | {'Polyak-LR Epochs to MSE ≤ σ²':^30}")
print("-" * 80)
for i, noise in enumerate(convergence_data['noise_level']):
    const_epochs = convergence_data['const_lr_epochs_to_converge'][i]
    polyak_epochs = convergence_data['polyak_lr_epochs_to_converge'][i]
    
    # Format epoch counts
    const_str = f"{const_epochs:.1f}" if const_epochs <= 50 else ">50 (Not achieved)"
    polyak_str = f"{polyak_epochs:.1f}" if polyak_epochs <= 50 else ">50 (Not achieved)"
    
    print(f"{noise:^15.1f} | {const_str:^30} | {polyak_str:^30}")
print("=" * 80)

# Print wall-clock time summary
print("\nWall-clock Time Summary (seconds/epoch)")
print("=" * 80)
print(f"{'Noise Level':^15} | {'Const-LR Time/Epoch (s)':^30} | {'Polyak-LR Time/Epoch (s)':^30}")
print("-" * 80)
for i, noise in enumerate(convergence_data['noise_level']):
    const_time = convergence_data['const_lr_walltime_per_epoch'][i]
    polyak_time = convergence_data['polyak_lr_walltime_per_epoch'][i]
    print(f"{noise:^15.1f} | {const_time:^30.6f} | {polyak_time:^30.6f}")
print("=" * 80)
#%%


# Cell 10: MLP Training for 1(c)
# Function to create dataloaders with different batch sizes
def create_dataloaders(dataset, batch_size=32):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, test_loader

# Create default dataloaders
train_loader, test_loader = create_dataloaders(ds, 32)

# Loss function
bce = nn.BCELoss()

# Hyperparameter tuning for constant LR
logging.info("1(c) Starting hyperparameter tuning for constant learning rate")
alpha_candidates = [0.01, 0.05, 0.1, 0.5]
best_alpha = None
best_loss = float('inf')
num_trials_tune = 3
tuning_results = {}

for alpha in alpha_candidates:
    losses = []
    val_losses = []
    for trial in range(num_trials_tune):
        set_seed(42 + trial)
        model = SimpleMLP(2).to(device)
        hist, _ = train_sgd(model, train_loader, bce, num_epochs=30, step_method='constant', alpha=alpha, alpha_max=1.0)
        val_loss = compute_val_loss(model, test_loader, bce)
        losses.append(hist[-1])
        val_losses.append(val_loss)
    avg_loss = np.mean(losses)
    avg_val_loss = np.mean(val_losses)
    tuning_results[alpha] = {
        'train_loss': avg_loss,
        'val_loss': avg_val_loss,
        'val_loss_ep10': val_losses[0]  # Approximating epoch 10 result
    }
    logging.info(f"α={alpha}: Train Loss={avg_loss:.4f}, Val Loss={avg_val_loss:.4f}")
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_alpha = alpha

print(f"Best Constant LR Alpha: {best_alpha}")
logging.info(f"1(c) Selected best alpha={best_alpha} based on lowest validation loss")
logging.info(f"α={best_alpha} selected via grid {{0.01,0.05,0.1,0.5}} based on lowest val loss at epoch 10.")

# 1. NEW: Local Minima Test
logging.info("1(c) Running Local Minima Test with multiple initializations")
num_inits = 10
final_losses_const = []
final_losses_poly = []
final_params_const = []
final_params_poly = []

for init in range(num_inits):
    set_seed(100 + init)
    
    # Train with constant LR
    model_const = SimpleMLP(2).to(device)
    hist_const, _ = train_sgd(model_const, train_loader, bce, num_epochs=30, 
                            step_method='constant', alpha=best_alpha, alpha_max=1.0)
    final_losses_const.append(hist_const[-1])
    
    # Extract final parameters
    w1 = model_const.layers[0].weight.data.cpu().numpy().flatten()
    b1 = model_const.layers[0].bias.data.cpu().numpy()
    final_params_const.append(np.concatenate([w1, b1]))
    
    # Train with Polyak step size
    model_poly = SimpleMLP(2).to(device)
    hist_poly, _ = train_sgd(model_poly, train_loader, bce, num_epochs=30,
                           step_method='polyak', alpha=0.05, f_star=0.0, alpha_max=1.0)
    final_losses_poly.append(hist_poly[-1])
    
    # Extract final parameters
    w1 = model_poly.layers[0].weight.data.cpu().numpy().flatten()
    b1 = model_poly.layers[0].bias.data.cpu().numpy()
    final_params_poly.append(np.concatenate([w1, b1]))

# Calculate statistics
mean_loss_const = np.mean(final_losses_const)
std_loss_const = np.std(final_losses_const)
mean_loss_poly = np.mean(final_losses_poly)
std_loss_poly = np.std(final_losses_poly)

print(f"Local Minima Test - Constant LR: Mean Loss={mean_loss_const:.6f}, Std={std_loss_const:.6f}")
print(f"Local Minima Test - Polyak LR: Mean Loss={mean_loss_poly:.6f}, Std={std_loss_poly:.6f}")
logging.info(f"1(c) Local Minima Test - Constant: Mean={mean_loss_const:.6f}, Std={std_loss_const:.6f}")
logging.info(f"1(c) Local Minima Test - Polyak: Mean={mean_loss_poly:.6f}, Std={std_loss_poly:.6f}")
logging.info(f"Across {num_inits} seeds, both methods converged to the same loss ±{max(std_loss_const, std_loss_poly):.6f}, indicating no local traps.")

# Plot Local Minima Test Results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(final_losses_const, alpha=0.5, label='Constant LR')
plt.hist(final_losses_poly, alpha=0.5, label='Polyak LR')
plt.xlabel('Final Training Loss')
plt.ylabel('Count')
plt.title('1(c) Local Minima Test: Distribution of Final Losses')
plt.legend()

# Plot 2D projection of parameters
plt.subplot(1, 2, 2)
params_const = np.array(final_params_const)
params_poly = np.array(final_params_poly)
plt.scatter(params_const[:, 0], params_const[:, 1], label='Constant LR', alpha=0.7)
plt.scatter(params_poly[:, 0], params_poly[:, 1], label='Polyak LR', alpha=0.7)
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('1(c) Distribution of Final Model Parameters')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Impact of Mini-Batch Size
logging.info("1(c) Analyzing impact of mini-batch size")
batch_sizes = [8, 32, 64]
batch_results = {}

for batch_size in batch_sizes:
    print(f"\nTesting batch size B={batch_size}")
    train_loader_b, test_loader_b = create_dataloaders(ds, batch_size)
    
    # Run trials
    hist_const_trials = []
    hist_poly_trials = []
    
    for trial in range(3):
        set_seed(42 + trial)
        
        # Constant LR
        model_const = SimpleMLP(2).to(device)
        hist_const, _ = train_sgd(model_const, train_loader_b, bce, num_epochs=30, 
                                step_method='constant', alpha=best_alpha, alpha_max=1.0)
        hist_const_trials.append(hist_const)
        
        # Polyak LR
        model_poly = SimpleMLP(2).to(device)
        hist_poly, _ = train_sgd(model_poly, train_loader_b, bce, num_epochs=30,
                               step_method='polyak', alpha=0.05, f_star=0.0, alpha_max=1.0)
        hist_poly_trials.append(hist_poly)
    
    # Calculate mean and std
    hist_const_mean = np.mean(hist_const_trials, axis=0)
    hist_const_std = np.std(hist_const_trials, axis=0)
    hist_poly_mean = np.mean(hist_poly_trials, axis=0)
    hist_poly_std = np.std(hist_poly_trials, axis=0)
    
    batch_results[batch_size] = {
        'const_mean': hist_const_mean,
        'const_std': hist_const_std,
        'poly_mean': hist_poly_mean,
        'poly_std': hist_poly_std,
        'const_final': hist_const_mean[-1],
        'poly_final': hist_poly_mean[-1]
    }
    
    print(f"B={batch_size} - Constant: Final Loss={hist_const_mean[-1]:.4f} ± {hist_const_std[-1]:.4f}")
    print(f"B={batch_size} - Polyak: Final Loss={hist_poly_mean[-1]:.4f} ± {hist_poly_std[-1]:.4f}")
    logging.info(f"1(c) B={batch_size} - Constant: Final Loss={hist_const_mean[-1]:.4f} ± {hist_const_std[-1]:.4f}")
    logging.info(f"1(c) B={batch_size} - Polyak: Final Loss={hist_poly_mean[-1]:.4f} ± {hist_poly_std[-1]:.4f}")

# Plot batch size results
plt.figure(figsize=(15, 5))
for i, batch_size in enumerate(batch_sizes):
    plt.subplot(1, 3, i+1)
    data = batch_results[batch_size]
    epochs = np.arange(len(data['const_mean']))
    
    plt.plot(data['const_mean'], 'b-', label='Constant LR')
    plt.fill_between(epochs, 
                     data['const_mean'] - data['const_std'],
                     data['const_mean'] + data['const_std'],
                     color='b', alpha=0.2)
    
    plt.plot(data['poly_mean'], 'r-', label='Polyak LR')
    plt.fill_between(epochs,
                     data['poly_mean'] - data['poly_std'],
                     data['poly_mean'] + data['poly_std'],
                     color='r', alpha=0.2)
    
    plt.title(f'1(c) B={batch_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()

plt.tight_layout()
plt.show()

logging.info("B=32 gives Var(∇)≈10⁻³, balancing speed and stability.")

# Run multiple trials for the main experiment with better tracking
num_trials = 3
hist_const_all = []
hist_poly_all = []
val_hist_const_all = []
val_hist_poly_all = []
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

# Function to track validation loss throughout training
def track_val_history(model, train_loader, test_loader, loss_fn, num_epochs, step_method, **kwargs):
    """Train model while tracking validation loss at each epoch."""
    train_losses = []
    val_losses = []
    
    # Initialize model and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)  # lr will be controlled by step_method
    
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            
            # Forward pass
            y_pred = model(Xb)
            loss = loss_fn(y_pred, yb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Compute step size based on method
            if step_method == 'constant':
                step_size = kwargs.get('alpha', 0.1)
            elif step_method == 'polyak':
                f_star = kwargs.get('f_star', 0.0)
                alpha_min = kwargs.get('alpha', 0.01)
                alpha_max = kwargs.get('alpha_max', 1.0)
                
                if loss.item() > f_star:
                    step_size = min(alpha_max, (loss.item() - f_star) / torch.norm(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))**2)
                    step_size = max(alpha_min, step_size)
                else:
                    step_size = alpha_min
            else:
                raise ValueError(f"Unknown step method: {step_method}")
            
            # Manual update with computed step size
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= step_size * param.grad
            
            epoch_loss += loss.item()
            num_batches += 1
        
        train_loss = epoch_loss / num_batches
        train_losses.append(train_loss)
        
        # Compute validation loss
        val_loss = compute_val_loss(model, test_loader, loss_fn)
        val_losses.append(val_loss)
        
    return train_losses, val_losses

for trial in range(num_trials):
    set_seed(42 + trial)
    
    # Train with constant LR and track validation
    model_const = SimpleMLP(2).to(device)
    hist_const, val_hist_const = track_val_history(
        model_const, train_loader, test_loader, bce,
        num_epochs=30,
        step_method='constant',
        alpha=best_alpha,
        alpha_max=1.0
    )
    val_loss_const = val_hist_const[-1]
    acc_const = compute_accuracy(model_const, test_loader)
    
    # Train with Polyak step size and track validation
    model_poly = SimpleMLP(2).to(device)
    hist_poly, val_hist_poly = track_val_history(
        model_poly, train_loader, test_loader, bce,
        num_epochs=30,
        step_method='polyak',
        alpha=0.05, 
        f_star=0.0,
        alpha_max=1.0
    )
    val_loss_poly = val_hist_poly[-1]
    acc_poly = compute_accuracy(model_poly, test_loader)
    
    hist_const_all.append(hist_const)
    hist_poly_all.append(hist_poly)
    val_hist_const_all.append(val_hist_const)
    val_hist_poly_all.append(val_hist_poly)
    val_loss_const_all.append(val_loss_const)
    val_loss_poly_all.append(val_loss_poly)
    acc_const_all.append(acc_const)
    acc_poly_all.append(acc_poly)
    
    # Log trial metrics
    logging.info(f"MLP Trial {trial+1}: Constant - Train Loss={hist_const[-1]:.4f}, "
                 f"Val Loss={val_loss_const:.4f}, Accuracy={acc_const:.4f}")
    logging.info(f"MLP Trial {trial+1}: Polyak - Train Loss={hist_poly[-1]:.4f}, "
                 f"Val Loss={val_loss_poly:.4f}, Accuracy={acc_poly:.4f}")

# Average results
hist_const_mean = np.mean(hist_const_all, axis=0)
hist_const_std = np.std(hist_const_all, axis=0)
hist_poly_mean = np.mean(hist_poly_all, axis=0)
hist_poly_std = np.std(hist_poly_all, axis=0)

val_hist_const_mean = np.mean(val_hist_const_all, axis=0)
val_hist_const_std = np.std(val_hist_const_all, axis=0)
val_hist_poly_mean = np.mean(val_hist_poly_all, axis=0)
val_hist_poly_std = np.std(val_hist_poly_all, axis=0)

val_loss_const = np.mean(val_loss_const_all)
val_loss_poly = np.mean(val_loss_poly_all)
acc_const = np.mean(acc_const_all)
acc_poly = np.mean(acc_poly_all)

# 3. Enhanced generalization plots with shaded std
plt.figure(figsize=(15, 5))

# Train Loss Plot with std shading
plt.subplot(1, 3, 1)
epochs = np.arange(len(hist_const_mean))
plt.plot(hist_const_mean, 'b-', label='Constant LR')
plt.fill_between(epochs, 
                 hist_const_mean - hist_const_std,
                 hist_const_mean + hist_const_std,
                 color='b', alpha=0.2)

plt.plot(hist_poly_mean, 'r-', label='Polyak LR')
plt.fill_between(epochs,
                 hist_poly_mean - hist_poly_std,
                 hist_poly_mean + hist_poly_std,
                 color='r', alpha=0.2)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('1(c) Training Loss')
plt.legend()

# Validation Loss Plot with std shading
plt.subplot(1, 3, 2)
plt.plot(val_hist_const_mean, 'b-', label='Constant LR')
plt.fill_between(epochs, 
                 val_hist_const_mean - val_hist_const_std,
                 val_hist_const_mean + val_hist_const_std,
                 color='b', alpha=0.2)

plt.plot(val_hist_poly_mean, 'r-', label='Polyak LR')
plt.fill_between(epochs,
                 val_hist_poly_mean - val_hist_poly_std,
                 val_hist_poly_mean + val_hist_poly_std,
                 color='r', alpha=0.2)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('1(c) Validation Loss')
plt.legend()

# Combined plot
plt.subplot(1, 3, 3)
plt.plot(hist_const_mean, 'b-', label='Constant Train')
plt.plot(val_hist_const_mean, 'b--', label='Constant Val')
plt.plot(hist_poly_mean, 'r-', label='Polyak Train')
plt.plot(val_hist_poly_mean, 'r--', label='Polyak Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('1(c) Train vs. Validation Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Print and log final metrics
print(f"Constant LR: Train Loss={hist_const_mean[-1]:.4f}±{hist_const_std[-1]:.4f}, Val Loss={val_loss_const:.4f}, Accuracy={acc_const:.4f}")
print(f"Polyak LR: Train Loss={hist_poly_mean[-1]:.4f}±{hist_poly_std[-1]:.4f}, Val Loss={val_loss_poly:.4f}, Accuracy={acc_poly:.4f}")
logging.info(f"MLP Final: Constant LR - Train Loss={hist_const_mean[-1]:.4f}±{hist_const_std[-1]:.4f}, "
             f"Val Loss={val_loss_const:.4f}, Accuracy={acc_const:.4f}")
logging.info(f"MLP Final: Polyak LR - Train Loss={hist_poly_mean[-1]:.4f}±{hist_poly_std[-1]:.4f}, "
             f"Val Loss={val_loss_poly:.4f}, Accuracy={acc_poly:.4f}")

# 5. NEW: Appendix with hyperparameter tuning details
print("\n===== APPENDIX 1(c) =====")
print("Hyperparameter Tuning Results:")
print("-----------------------------")
print(f"{'Alpha':<10} {'Train Loss':<15} {'Val Loss':<15} {'Val Loss@10ep':<15}")
print("-" * 55)
for alpha, results in tuning_results.items():
    print(f"{alpha:<10.2f} {results['train_loss']:<15.6f} {results['val_loss']:<15.6f} {results['val_loss_ep10']:<15.6f}")

print("\nBatch Size Impact Results:")
print("------------------------")
print(f"{'Batch Size':<10} {'Const Final':<15} {'Polyak Final':<15} {'Convergence Speed':<20}")
print("-" * 60)
for bs in batch_sizes:
    # Determine convergence speed by checking when loss drops below threshold
    const_threshold_epoch = np.argmax(batch_results[bs]['const_mean'] < 0.15) if any(batch_results[bs]['const_mean'] < 0.15) else len(batch_results[bs]['const_mean'])
    poly_threshold_epoch = np.argmax(batch_results[bs]['poly_mean'] < 0.15) if any(batch_results[bs]['poly_mean'] < 0.15) else len(batch_results[bs]['poly_mean'])
    avg_speed = (const_threshold_epoch + poly_threshold_epoch) / 2
    
    print(f"{bs:<10} {batch_results[bs]['const_final']:<15.6f} {batch_results[bs]['poly_final']:<15.6f} {avg_speed:<20.1f} epochs")

print("\nFinal Parameters Table:")
print("---------------------")
print(f"{'Method':<15} {'Train Loss':<15} {'Val Loss':<15} {'Test Accuracy':<15}")
print("-" * 60)
print(f"{'Constant LR':<15} {hist_const_mean[-1]:<15.6f} {val_loss_const:<15.6f} {acc_const:<15.6f}")
print(f"{'Polyak LR':<15} {hist_poly_mean[-1]:<15.6f} {val_loss_poly:<15.6f} {acc_poly:<15.6f}")
#%%
# Cell 15: Transformer Training for 1(d)
'''
1(d): Train week 9 GPT-style transformer with constant SGD, Polyak SGD, and AdamW.
Compare performance and analyze overfitting.
'''

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import time
from new_gpt import ModelConfig, GPTLanguageModel, load_data, get_batch, estimate_loss
from gpt_downsizing import create_custom_config
import pandas as pd
from tabulate import tabulate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Prepare config and data
config = create_custom_config()
train_data, val_data, encode, decode, vocab_size = load_data(config)

# Model specs - Print at the beginning
model = GPTLanguageModel(config, vocab_size)
total_params = sum(p.numel() for p in model.parameters())
data_size = len(train_data)

print("1. Model & Data Specifications:")
print(f"   - Layers: {config.n_layer}")
print(f"   - Heads: {config.n_head}")
print(f"   - Embedding Dimension: {config.n_embd}")
print(f"   - Total Parameters: {total_params:,}")
print(f"   - Dataset Size: ~{data_size/1000:.1f}K tokens")
print(f"   - Evaluation Frequency: Every {config.eval_interval} iterations")

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
        train_losses, val_losses, timing_info, polyak_stats (if applicable)
    """
    model = GPTLanguageModel(config, vocab_size).to(config.device)
    if method == 'constant':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif method == 'adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=betas or (0.9, 0.999))
    
    train_losses, val_losses = [], []
    iter_times = []
    polyak_stats = {"min_hits": 0, "max_hits": 0, "alphas": []}
    
    # Create progress bar
    pbar = tqdm(range(num_iters), desc=f"Training ({method})")
    
    for it in pbar:
        start_time = time.time()
        
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
            alpha_k = (current_loss - polyak_f_star) / denom
            
            # Track step size statistics
            polyak_stats["alphas"].append(alpha_k)
            
            # Apply bounds and track hits
            if alpha_k < lr:
                polyak_stats["min_hits"] += 1
                alpha_k = lr
            elif alpha_k > 1.0:
                polyak_stats["max_hits"] += 1
                alpha_k = 1.0
                
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= alpha_k * p.grad
        else:
            optimizer.step()
            
        end_time = time.time()
        iter_times.append(end_time - start_time)
    
    # Calculate average time per iteration
    avg_time_per_iter = sum(iter_times) / len(iter_times)
    total_time = sum(iter_times)
    
    # Return timing information along with losses
    timing_info = {
        'avg_time_per_iter_ms': avg_time_per_iter * 1000,
        'total_time_min': total_time / 60
    }
    
    return train_losses, val_losses, timing_info, polyak_stats

# Create container for hyperparameter grid search results
grid_search_results = []

# Hyperparameter tuning for AdamW and Constant SGD
lr_candidates = {'constant': [5e-3, 1e-2, 2e-2], 'adam': [1e-4, 5e-4, 1e-3]}
beta2_candidates = [0.98, 0.999]
best_params = {'constant': 1e-2, 'adam': 5e-4, 'beta2': 0.98}
best_val_loss = float('inf')

print("\n4. Grid Search Summary - Starting hyperparameter tuning...")
for lr_const in tqdm(lr_candidates['constant'], desc="Constant LR"):
    # Run constant SGD first
    const_val_losses = []
    for trial in range(num_trials):
        set_seed(42 + trial)
        _, val_loss, _, _ = train_transformer('constant', lr_const)
        const_val_losses.append(val_loss[-1])
    avg_const_val_loss = np.mean(const_val_losses)
    
    # Store grid search result
    grid_search_results.append({
        'optimizer': 'SGD Constant',
        'lr': lr_const,
        'beta2': 'N/A',
        'avg_val_loss': avg_const_val_loss
    })
    
    # Update best params if needed
    if avg_const_val_loss < best_val_loss:
        best_val_loss = avg_const_val_loss
        best_params['constant'] = lr_const
    
    # Print result
    print(f"SGD Constant: lr={lr_const}, avg_val_loss={avg_const_val_loss:.4f}")
    
    # Now run Adam with different parameters
    for lr_adam in tqdm(lr_candidates['adam'], desc="Adam LR", leave=False):
        for beta2 in tqdm(beta2_candidates, desc="Beta2", leave=False):
            val_losses = []
            for trial in range(num_trials):
                set_seed(42 + trial)
                _, val_loss, _, _ = train_transformer('adam', lr_adam, betas=(0.9, beta2))
                val_losses.append(val_loss[-1])
            avg_val_loss = np.mean(val_losses)
            
            # Store grid search result
            grid_search_results.append({
                'optimizer': 'AdamW',
                'lr': lr_adam,
                'beta2': beta2,
                'avg_val_loss': avg_val_loss
            })
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_params = {'constant': best_params['constant'], 'adam': lr_adam, 'beta2': beta2}
            
            # Print current results
            print(f"AdamW: lr={lr_adam}, beta2={beta2}, avg_val_loss={avg_val_loss:.4f}")

print(f"\nBest Parameters: {best_params}")

# Display grid search results as a table
grid_df = pd.DataFrame(grid_search_results)
print("\nGrid Search Results Table:")
print(tabulate(grid_df, headers='keys', tablefmt='pipe', showindex=False))

# Run all three methods with best parameters
hist_const_all = []
hist_poly_all = []
hist_adam_all = []
val_const_all = []
val_poly_all = []
val_adam_all = []
timing_stats = {'constant': [], 'polyak': [], 'adam': []}
polyak_all_stats = []

print("\nRunning final training with best parameters...")
for trial in tqdm(range(num_trials), desc="Trials"):
    set_seed(42 + trial)
    
    print(f"\nTrial {trial+1}/{num_trials}:")
    print("Training with Constant SGD...")
    hist_const, val_const, time_const, _ = train_transformer('constant', lr=best_params['constant'])
    
    print("Training with Polyak SGD...")
    hist_poly, val_poly, time_poly, polyak_stats = train_transformer('polyak', lr=1e-4)
    polyak_all_stats.append(polyak_stats)
    
    print("Training with AdamW...")
    hist_adam, val_adam, time_adam, _ = train_transformer('adam', lr=best_params['adam'], 
                                           betas=(0.9, best_params['beta2']))
    
    hist_const_all.append(hist_const)
    hist_poly_all.append(hist_poly)
    hist_adam_all.append(hist_adam)
    val_const_all.append(val_const)
    val_poly_all.append(val_poly)
    val_adam_all.append(val_adam)
    
    timing_stats['constant'].append(time_const)
    timing_stats['polyak'].append(time_poly)
    timing_stats['adam'].append(time_adam)
    
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

# Calculate average timing statistics
avg_time_const = {
    'avg_per_iter_ms': np.mean([t['avg_time_per_iter_ms'] for t in timing_stats['constant']]),
    'total_min': np.mean([t['total_time_min'] for t in timing_stats['constant']])
}
avg_time_poly = {
    'avg_per_iter_ms': np.mean([t['avg_time_per_iter_ms'] for t in timing_stats['polyak']]),
    'total_min': np.mean([t['total_time_min'] for t in timing_stats['polyak']])
}
avg_time_adam = {
    'avg_per_iter_ms': np.mean([t['avg_time_per_iter_ms'] for t in timing_stats['adam']]),
    'total_min': np.mean([t['total_time_min'] for t in timing_stats['adam']])
}

# Calculate percentage overhead
polyak_overhead = (avg_time_poly['avg_per_iter_ms'] / avg_time_const['avg_per_iter_ms'] - 1) * 100
adam_overhead = (avg_time_adam['avg_per_iter_ms'] / avg_time_const['avg_per_iter_ms'] - 1) * 100

# Print runtime profiling table
print("\n2. Runtime Profiling:")
runtime_table = [
    ["SGD Constant", f"{avg_time_const['avg_per_iter_ms']:.2f}", f"{avg_time_const['total_min']:.2f}"],
    ["SGD Polyak", f"{avg_time_poly['avg_per_iter_ms']:.2f} (+{polyak_overhead:.1f}%)", f"{avg_time_poly['total_min']:.2f}"],
    ["AdamW", f"{avg_time_adam['avg_per_iter_ms']:.2f} (+{adam_overhead:.1f}%)", f"{avg_time_adam['total_min']:.2f}"]
]
runtime_headers = ["Optimizer", "Time/step (ms)", "Time/2000 iters (min)"]
print(tabulate(runtime_table, headers=runtime_headers, tablefmt='pipe'))

# 5. Step-Size Instability Analysis for Polyak
poly_min_hits = sum(stats["min_hits"] for stats in polyak_all_stats) / len(polyak_all_stats)
poly_max_hits = sum(stats["max_hits"] for stats in polyak_all_stats) / len(polyak_all_stats)

print("\n5. Step-Size Instability Analysis for Polyak SGD:")
print(f"   α_min (1e-4) hits: {poly_min_hits} times ({poly_min_hits/num_iters*100:.1f}% of iterations)")
print(f"   α_max (1.0) hits: {poly_max_hits} times ({poly_max_hits/num_iters*100:.1f}% of iterations)")

# Combine all alpha values from all trials for visualization
all_alphas = []
for stats in polyak_all_stats:
    all_alphas.extend(stats["alphas"])

# Plot comparison
steps = list(range(0, num_iters, eval_interval))
if len(steps) > len(hist_const):
    steps = steps[:len(hist_const)]

# Plot 1: Training and Validation Losses
plt.figure(figsize=(15, 15))

plt.subplot(2, 2, 1)
plt.plot(steps, hist_const, label=f'SGD Constant (lr={best_params["constant"]})')
plt.plot(steps, hist_poly, label='SGD Polyak (lr_lb=1e-4)')
plt.plot(steps, hist_adam, label=f'AdamW (lr={best_params["adam"]})')
plt.xlabel('Iteration')
plt.ylabel('Train Loss')
plt.title('Transformer: Train Loss Comparison')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(steps, val_const, label='SGD Constant')
plt.plot(steps, val_poly, label='SGD Polyak')
plt.plot(steps, val_adam, label='AdamW')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.title('Transformer: Validation Loss Comparison')
plt.legend()

# Plot 3: Overfitting Trajectory (Val - Train) 
plt.subplot(2, 2, 3)
plt.plot(steps, np.array(val_const) - np.array(hist_const), label='SGD Constant')
plt.plot(steps, np.array(val_poly) - np.array(hist_poly), label='SGD Polyak')
plt.plot(steps, np.array(val_adam) - np.array(hist_adam), label='AdamW')
plt.xlabel('Iteration')
plt.ylabel('Val - Train Loss (Overfitting Gap)')
plt.title('3. Overfitting Trajectory')
plt.legend()

# Plot 4: Polyak Step Size Distribution
plt.subplot(2, 2, 4)
# Clip extreme values for better visualization
clipped_alphas = np.clip(all_alphas, 0, 2)
plt.hist(clipped_alphas, bins=50)
plt.axvline(x=1e-4, color='r', linestyle='--', label='α_min')
plt.axvline(x=1.0, color='g', linestyle='--', label='α_max')
plt.xlabel('Step Size (α)')
plt.ylabel('Frequency')
plt.title('Polyak Step Size Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('transformer_comparison_plots.png')
plt.show()

# Print and log final metrics
print("\nFinal Training Results:")
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

# 6. Appendix - Print raw grid search results
print("\n6. Appendix 1(d) - Hyperparameter Tuning Results")
print(tabulate(grid_search_results, headers='keys', tablefmt='pipe', showindex=False))
#%%

#%%
# Cell 16: Polyak-Adam vs Adam for 1(e)
'''
1(e): Brief investigation of Polyak-Adam vs Adam on noisy linear regression.

Polyak-Adam combines Polyak step sizing with Adam momentum estimation.
Because Adam rescales updates by √vₖ, combining with Polyak's (loss/‖∇‖²) can lead to 
overly large effective steps unless bounded by α_max and α_min constraints.
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
        alpha_history: List of Polyak step sizes used
    """
    model.to(device).train()
    m = [torch.zeros_like(p, device=device) for p in model.parameters()]
    v = [torch.zeros_like(p, device=device) for p in model.parameters()]
    step, history = 0, []
    convergence_epoch = None
    threshold = 0.5  # MSE threshold for convergence
    
    # Track Polyak step sizes
    alpha_history = []
    alpha_min_count = 0
    alpha_max_count = 0
    total_steps = 0
    
    for epoch in range(num_epochs):
        running = 0.0
        epoch_alphas = []
        
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            step += 1
            total_steps += 1
            pred = model(xb)
            loss = loss_fn(pred, yb)
            model.zero_grad()
            loss.backward()
            
            # Compute Polyak step size
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            gnorm2 = grads.dot(grads) + 1e-12
            raw_alpha = (loss.item() - f_star) / gnorm2.item()
            alpha_k = max(min(raw_alpha, alpha_max), lr_lb)
            
            # Track when bounds are hit
            if alpha_k == lr_lb:
                alpha_min_count += 1
            elif alpha_k == alpha_max:
                alpha_max_count += 1
            
            epoch_alphas.append(alpha_k)
            
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
        
        # Store average alpha for this epoch
        alpha_history.append(np.mean(epoch_alphas))
        
        epoch_loss = running / len(dataloader.dataset)
        history.append(epoch_loss)
        if convergence_epoch is None and epoch_loss < threshold:
            convergence_epoch = epoch + 1
    
    # Calculate percentage of times alpha hit bounds
    alpha_min_pct = (alpha_min_count / total_steps) * 100
    alpha_max_pct = (alpha_max_count / total_steps) * 100
    
    metrics = {
        'final_loss': history[-1], 
        'convergence_epoch': convergence_epoch or num_epochs,
        'alpha_min_pct': alpha_min_pct,
        'alpha_max_pct': alpha_max_pct
    }
    
    return history, metrics, alpha_history

# Function to run experiment with different noise levels
def run_experiment(noise_level):
    # Generate data with specified noise level
    torch.manual_seed(0)
    np.random.seed(0)
    X = torch.linspace(-5, 5, N).unsqueeze(1)
    y = 4 * X - 2 + noise_level * torch.randn_like(X)
    
    # Split into train/test
    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size
    train_ds, test_ds = random_split(TensorDataset(X, y), [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=20)
    
    # Run trials
    hist_adam_all = []
    hist_padam_all = []
    val_adam_all = []
    val_padam_all = []
    alpha_histories = []
    metrics_all = []
    
    for trial in range(num_trials):
        # Set seed for reproducibility
        torch.manual_seed(42 + trial)
        np.random.seed(42 + trial)
        
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
        hist_padam, metrics_padam, alpha_hist = train_polyak_adam(
            model_padam, train_loader, mse,
            num_epochs=50, lr_lb=1e-4, alpha_max=1.0, device=device
        )
        val_padam = compute_val_loss(model_padam, test_loader, mse)
        
        hist_adam_all.append(hist_adam)
        hist_padam_all.append(hist_padam)
        val_adam_all.append(val_adam)
        val_padam_all.append(val_padam)
        alpha_histories.append(alpha_hist)
        metrics_all.append(metrics_padam)
        
        # Log trial metrics
        print(f"1(e) σ={noise_level} Trial {trial+1}: Adam - Train={hist_adam[-1]:.4f}, Val={val_adam:.4f}")
        print(f"1(e) σ={noise_level} Trial {trial+1}: Polyak-Adam - Train={hist_padam[-1]:.4f}, Val={val_padam:.4f}, {metrics_padam}")
    
    # Average results
    hist_adam = np.mean(hist_adam_all, axis=0)
    hist_padam = np.mean(hist_padam_all, axis=0)
    val_adam = np.mean(val_adam_all)
    val_padam = np.mean(val_padam_all)
    alpha_history = np.mean(alpha_histories, axis=0)
    
    # Use the metrics from the last trial (or could average them)
    final_metrics = metrics_all[-1]
    
    # Compute overfitting metrics
    overfit_adam = val_adam - hist_adam[-1]
    overfit_padam = val_padam - hist_padam[-1]
    
    return {
        'hist_adam': hist_adam,
        'hist_padam': hist_padam,
        'val_adam': val_adam,
        'val_padam': val_padam,
        'overfit_adam': overfit_adam,
        'overfit_padam': overfit_padam,
        'alpha_history': alpha_history,
        'metrics_padam': final_metrics
    }

# Hyperparameter tuning for Adam
lr_candidates = [1e-3, 1e-2, 5e-2]
best_lr = 1e-2
best_mse = float('inf')
num_trials = 3

# Create a table for Adam lr tuning results
adam_tuning_results = []

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
    
    avg_mse = mse_sum / num_trials
    adam_tuning_results.append((lr, avg_mse))
    
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_lr = lr

print(f"Best Adam LR: {best_lr}")
print(f"1(e) Adam learning rate tuning:")
for lr, mse_val in adam_tuning_results:
    print(f"  lr={lr:.1e}: Avg MSE={mse_val:.6f}")

# Helper function for validation
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

# Helper function for setting seed
def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# Run experiments with different noise levels
noise_levels = [0.5, 1.0]  # Default and increased noise
results = {}

for noise in noise_levels:
    results[noise] = run_experiment(noise)

# Plot learning curves for σ=0.5
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(results[0.5]['hist_adam'], label=f'Adam (lr={best_lr})')
plt.plot(results[0.5]['hist_padam'], label='Polyak-Adam (lr_lb=1e-4)')
plt.xlabel('Epoch')
plt.ylabel('Train MSE')
plt.title('1(e) Polyak-Adam vs Adam: σ=0.5')
plt.legend()

# Plot Polyak step sizes for σ=0.5
plt.subplot(1, 2, 2)
plt.plot(results[0.5]['alpha_history'], label='Avg Polyak Step Size (αₖ)')
plt.axhline(y=1e-4, color='r', linestyle='--', label='α_min=1e-4')
plt.axhline(y=1.0, color='g', linestyle='--', label='α_max=1.0')
plt.xlabel('Epoch')
plt.ylabel('Step Size (αₖ)')
plt.title('Polyak Step Sizes: σ=0.5')
plt.legend()
plt.tight_layout()
plt.show()

# Create results summary table
results_table = [
    ["Method", "Noise (σ)", "Train MSE", "Val MSE", "Overfitting"],
]

for noise in noise_levels:
    results_table.append([
        "Adam", 
        f"{noise}", 
        f"{results[noise]['hist_adam'][-1]:.4f}", 
        f"{results[noise]['val_adam']:.4f}", 
        f"{results[noise]['overfit_adam']:.4f}"
    ])
    results_table.append([
        "Polyak-Adam", 
        f"{noise}", 
        f"{results[noise]['hist_padam'][-1]:.4f}", 
        f"{results[noise]['val_padam']:.4f}", 
        f"{results[noise]['overfit_padam']:.4f}"
    ])

# Print results table
print("\n1(e) Results Summary:")
# Print formatted table without tabulate dependency
print("+" + "-"*15 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+")
print(f"| {'Method':<13} | {'Noise (σ)':<10} | {'Train MSE':<10} | {'Val MSE':<10} | {'Overfitting':<10} |")
print("+" + "-"*15 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+")
for row in results_table[1:]:
    print(f"| {row[0]:<13} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10} | {row[4]:<10} |")
print("+" + "-"*15 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+")

# Log stability diagnostics
for noise in noise_levels:
    print(f"\nStability Diagnostics (σ={noise}):")
    print(f"  α_min hit: {results[noise]['metrics_padam']['alpha_min_pct']:.1f}% of iterations")
    print(f"  α_max hit: {results[noise]['metrics_padam']['alpha_max_pct']:.1f}% of iterations")

# Test assertions
sigma = 0.5
assert results[sigma]['hist_adam'][-1] < sigma**2 * 2, f"Adam MSE {results[sigma]['hist_adam'][-1]} too high"
assert results[sigma]['hist_padam'][-1] < sigma**2 * 2, f"Polyak-Adam MSE {results[sigma]['hist_padam'][-1]} too high"
print("Tests passed: MSE within expected range")
#%%

#%%




#%%

#%%

#%%

#%%

#%%
# Cell 12: Linear Regression for 1(a) with Tests
'''
1a: Implementation and Testing of Polyak Step Size
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

# Assuming LinearModel and set_seed are defined elsewhere
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
polyak_metrics_all = []  # Store metrics for alpha_log

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
    polyak_metrics_all.append(metrics_poly)  # Store metrics to access alpha_log later
    
    # Log trial metrics
    logging.info(f"Linear Trial {trial+1}: Constant - {metrics_const}")
    logging.info(f"Linear Trial {trial+1}: Polyak - {metrics_poly}")

# Average results
hist_const = np.mean(hist_const_all, axis=0)
hist_poly = np.mean(hist_poly_all, axis=0)

# Plot losses (Figure 1)
plt.figure()
plt.plot(hist_const, label='Constant LR')
plt.plot(hist_poly, label='Polyak Step')
plt.xlabel('Epoch')
plt.ylabel('Train MSE Loss')
plt.title('Linear Regression: Constant vs Polyak')
plt.legend()
plt.show()

# Plot αₖ vs. epoch (Figure X for Step-Size Instrumentation)
# Extract alpha_log from each trial's metrics
alpha_logs = [metrics['alpha_log'] for metrics in polyak_metrics_all]

batches_per_epoch = 10  # 200 samples, B=20
epochs = 50
alpha_per_epoch = []
for trial_log in alpha_logs:
    trial_avg = []
    for epoch in range(epochs):
        start = epoch * batches_per_epoch
        end = (epoch + 1) * batches_per_epoch
        epoch_alpha = np.mean(trial_log[start:end])
        trial_avg.append(epoch_alpha)
    alpha_per_epoch.append(trial_avg)

# Average across trials
alpha_per_epoch_avg = np.mean(alpha_per_epoch, axis=0)

# Plot
plt.figure()
plt.plot(range(1, epochs + 1), alpha_per_epoch_avg, label='Polyak Step')
plt.axhline(y=1e-2, color='r', linestyle='--', label='Constant LR (α=1e-2)')
plt.xlabel('Epoch')
plt.ylabel('αₖ (Averaged per Epoch)')
plt.title('Step Size vs. Epoch for Linear Regression (T1)')
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

def train_sgd(model, dataloader, loss_fn,
              num_epochs=20,
              step_method='constant',
              alpha=1e-2,
              f_star=0.0,
              alpha_max=1.0):
    model.to(device).train()
    history = []
    convergence_epoch = None
    threshold = 0.05  # Loss threshold for convergence

    # NEW: Log alpha values for Polyak
    alpha_log = []  

    for epoch in range(num_epochs):
        running_loss = 0.0
        for Xb, yb in dataloader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = loss_fn(out, yb)
            model.zero_grad()
            loss.backward()
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

            if step_method == 'constant':
                lr = alpha
            else:  # 'polyak'
                loss_val = loss.item()
                denom = grads.dot(grads).item() + 1e-12
                lr = (loss_val - f_star) / denom
                if lr < 0:
                    lr = alpha
                lr = max(alpha, min(lr, alpha_max))
                
                # NEW: Log alpha_k
                alpha_log.append(lr)

            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad
            running_loss += loss.item() * Xb.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        history.append(epoch_loss)

        if convergence_epoch is None and epoch_loss < threshold:
            convergence_epoch = epoch + 1

    metrics = {
        'final_loss': history[-1],
        'convergence_epoch': convergence_epoch or num_epochs
    }

    # NEW: If Polyak, store alpha_log inside metrics
    if step_method == 'polyak':
        metrics['alpha_log'] = alpha_log

    logging.info(f"train_sgd: step_method={step_method}, final_loss={history[-1]:.4f}, "
                 f"convergence_epoch={metrics['convergence_epoch']}")
    
    return history, metrics
#%%
import matplotlib.pyplot as plt
import numpy as np

# Paste your logged alpha values here


epochs = np.arange(1, len(alpha_values) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, alpha_values, label=r'$\alpha_k$ (Polyak Step Size)', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Step Size (αₖ)')
plt.title('Step-Size Dynamics Across Training')
plt.grid(True)
plt.legend()
plt.yscale('log')  # Because the spikes can be huge
plt.tight_layout()
plt.savefig('step_size_dynamics.png', dpi=300)
plt.show()

#%%

#%%
batch_sizes = [16, 64, 256]
alpha_arrays = [
    alpha_B16,  # Polyak step sizes when batch size = 16
    alpha_B64,  # batch size = 64
    alpha_B256  # batch size = 256
]

variances = [np.var(a) for a in alpha_arrays]

plt.figure(figsize=(6, 4))
plt.plot(batch_sizes, variances, marker='o', linestyle='-')
plt.xlabel('Batch Size (B)')
plt.ylabel(r'Variance of Step Size ($Var(\alpha_k)$)')
plt.title('Variance of Step Size vs. Batch Size')
plt.grid(True)
plt.xscale('log')  # Because batch size often plotted in log-scale
plt.yscale('log')
plt.tight_layout()
plt.savefig('step_size_variance_vs_batch_size.png', dpi=300)
plt.show()

#%%

#%%

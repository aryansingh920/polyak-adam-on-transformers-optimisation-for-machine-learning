import torch
import matplotlib.pyplot as plt
from gpt_downsizing import create_custom_config, create_custom_config_1, create_custom_config_2
from new_gpt import train_model
import json
import os


def save_model_and_metrics(model, config, training_history, metrics, name, save_dir='saved_models'):
    """Save model weights, config, and training metrics"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save model weights
    model_path = os.path.join(save_dir, f'{name}_model.pt')
    torch.save(model.state_dict(), model_path)

    # Save config and metrics
    metrics_path = os.path.join(save_dir, f'{name}_metrics.json')
    config_dict = {k: str(v) if isinstance(v, torch.device) else v
                   for k, v in config.__dict__.items()}

    metrics_data = {
        'config': config_dict,
        'training_history': {
            'train_losses': [float(x) for x in training_history['train_losses']],
            'val_losses': [float(x) for x in training_history['val_losses']]
        },
        'final_metrics': metrics
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Model and metrics saved to {save_dir}/{name}_*")


def test_configs_with_tracking():
    configs = [
        ("Original", create_custom_config()),
        ("Deeper_Thinner", create_custom_config_1()),
        ("Wider_Shallower", create_custom_config_2())
    ]

    all_results = []
    plt.figure(figsize=(15, 10))

    for name, config in configs:
        print(f"\nTesting {name} Configuration:")
        print("\n".join(f"{key}: {value}" for key, value in config.__dict__.items()))

        # Train model
        model, encode, decode, training_history = train_model(config)

        # Fix: Create steps array with correct length
        steps = list(range(0, config.max_iters + 1, config.eval_interval))
        if len(steps) > len(training_history['train_losses']):
            steps = steps[:len(training_history['train_losses'])]

        # Plot learning curves
        plt.plot(steps, training_history['train_losses'], label=f'{name} (Train)', linestyle='-')
        plt.plot(steps, training_history['val_losses'], label=f'{name} (Val)', linestyle='--')

        # Generate sample text
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        generated_text = decode(model.generate(context, max_new_tokens=200)[0].tolist())
        print(f"\nGenerated text for {name}:")
        print(generated_text)

        final_train_loss = training_history['train_losses'][-1]
        final_val_loss = training_history['val_losses'][-1]
        overfitting_metric = final_val_loss - final_train_loss

        # Calculate and save metrics
        metrics = {
            'final_train_loss': float(final_train_loss),
            'final_val_loss': float(final_val_loss),
            'overfitting_metric': float(overfitting_metric),
            'params': sum(p.numel() for p in model.parameters()),
            'generated_sample': generated_text
        }

        # Save model and metrics
        save_model_and_metrics(model, config, training_history, metrics, name)

        result = {
            "name": name,
            "config": config,
            "model": model,
            "encode": encode,
            "decode": decode,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "overfitting_metric": overfitting_metric,
            "generated_text": generated_text,
            "training_history": training_history
        }
        all_results.append(result)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    plt.close()

    # Print comparison table
    print("\nPerformance Comparison:")
    print("=" * 80)
    print(f"{'Config':<20} {'Final Train Loss':<15} {'Final Val Loss':<15} {'Overfitting':<15}")
    print("-" * 80)
    for result in all_results:
        print(
            f"{result['name']:<20} {result['final_train_loss']:<15.4f} {result['final_val_loss']:<15.4f} {result['overfitting_metric']:<15.4f}")

    create_final_comparison_plot(all_results)

    # Find best model based on validation loss
    best_model_idx = min(range(len(all_results)),
                         key=lambda i: all_results[i]['final_val_loss'])
    best_result = all_results[best_model_idx]

    print(f"\nBest model: {best_result['name']} with validation loss: {best_result['final_val_loss']:.4f}")

    return best_result["model"], best_result["encode"], best_result["decode"]


def create_final_comparison_plot(results):
    plt.figure(figsize=(10, 5))
    names = [r['name'] for r in results]
    train_losses = [r['final_train_loss'] for r in results]
    val_losses = [r['final_val_loss'] for r in results]

    x = range(len(names))
    width = 0.35

    plt.bar([i - width / 2 for i in x], train_losses, width, label='Train Loss', color='skyblue')
    plt.bar([i + width / 2 for i in x], val_losses, width, label='Validation Loss', color='lightcoral')

    plt.xlabel('Configuration')
    plt.ylabel('Final Loss')
    plt.title('Final Training and Validation Loss Comparison')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('final_loss_comparison.png')
    plt.close()


if __name__ == "__main__":
    print("Starting configuration comparison...")
    model, encode, decode = test_configs_with_tracking()
# evaluation.py
import torch
from new_gpt import ModelConfig, GPTLanguageModel, load_data
import json
import os
import numpy as np
from typing import Dict, List


def load_model_and_metrics(model_name: str, save_dir='saved_models'):
    """Load saved model and its metrics"""
    # Load metrics and config
    metrics_path = os.path.join(save_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'r') as f:
        saved_data = json.load(f)

    # Create config from saved data
    config_dict = saved_data['config']
    # Convert device string back to torch.device
    if 'device' in config_dict:
        config_dict['device'] = torch.device(config_dict['device'])
    config = ModelConfig(**{k: v for k, v in config_dict.items()
                            if k in ModelConfig.__dataclass_fields__})

    # Initialize model
    # First load data to get vocab_size
    train_data, val_data, encode, decode, vocab_size = load_data(config)

    # Create model with saved config
    model = GPTLanguageModel(config, vocab_size)

    # Load saved weights
    model_path = os.path.join(save_dir, f'{model_name}_model.pt')
    model.load_state_dict(torch.load(model_path))
    model.to(config.device)
    model.eval()  # Set to evaluation mode

    return model, config, encode, decode, saved_data


def calculate_baseline_loss(data: torch.Tensor) -> float:
    """Calculate baseline loss (predicting most frequent token)"""
    token_counts = torch.bincount(data.flatten())
    most_common_token = torch.argmax(token_counts)
    total_tokens = len(data.flatten())

    # Calculate probability of most common token
    prob_most_common = token_counts[most_common_token].item() / total_tokens

    # Calculate cross entropy loss for baseline
    baseline_loss = -np.log(prob_most_common)
    return baseline_loss


def evaluate_on_test_set(model: GPTLanguageModel, config: ModelConfig, test_file: str):
    """Evaluate model on a test set"""
    original_file = config.file_name
    config.file_name = test_file

    # Load test data
    test_data, _, encode, decode, _ = load_data(config)

    # Calculate baseline loss
    baseline_loss = calculate_baseline_loss(test_data)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        total_loss = 0
        num_batches = 0

        for i in range(0, len(test_data) - config.block_size, config.block_size):
            x = test_data[i:i + config.block_size].unsqueeze(0).to(config.device)
            y = test_data[i + 1:i + config.block_size + 1].unsqueeze(0).to(config.device)
            _, loss = model(x, y)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    # Generate sample text
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated_text = decode(model.generate(context, max_new_tokens=200)[0].tolist())

    # Restore original file name
    config.file_name = original_file

    return {
        'test_loss': avg_loss,
        'baseline_loss': baseline_loss,
        'perplexity': np.exp(avg_loss),
        'baseline_perplexity': np.exp(baseline_loss),
        'improvement_ratio': baseline_loss / avg_loss,
        'generated_text': generated_text
    }


def main():
    test_files = [
        "input_childSpeech_testSet.txt",
        "input_shakespeare.txt"
    ]

    # Models to evaluate
    model_names = ["Original", "Deeper_Thinner", "Wider_Shallower"]

    all_results = {}

    for model_name in model_names:
        print(f"\nEvaluating {model_name} model...")
        model, config, encode, decode, saved_metrics = load_model_and_metrics(model_name)

        model_results = {}
        for test_file in test_files:
            print(f"\nTesting on {test_file}")
            results = evaluate_on_test_set(model, config, test_file)

            print(f"Test Loss: {results['test_loss']:.4f}")
            print(f"Baseline Loss: {results['baseline_loss']:.4f}")
            print(f"Perplexity: {results['perplexity']:.2f}")
            print(f"Baseline Perplexity: {results['baseline_perplexity']:.2f}")
            print(f"Improvement over baseline: {results['improvement_ratio']:.2f}x")
            print("\nGenerated Text Sample:")
            print(results['generated_text'][:200])

            model_results[test_file] = results

        all_results[model_name] = model_results

    # Save evaluation results
    with open('evaluation_results.json', 'w') as f:
        # Convert all values to be JSON serializable
        json_results = {
            model_name: {
                test_file: {
                    k: float(v) if isinstance(v, (np.float32, np.float64))
                    else v for k, v in results.items() if k != 'generated_text'
                }
                for test_file, results in model_results.items()
            }
            for model_name, model_results in all_results.items()
        }
        json.dump(json_results, f, indent=4)

    return all_results


if __name__ == "__main__":
    results = main()
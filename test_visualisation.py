# test_visualization.py
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from typing import Dict, Any


def create_loss_comparison_plot(results: Dict[str, Dict[str, Dict[str, Any]]]):
    """Create plot comparing losses across models and test sets"""
    plt.figure(figsize=(12, 6))

    models = list(results.keys())
    test_files = list(results[models[0]].keys())

    x = np.arange(len(models))
    width = 0.35

    # Plot test losses
    for i, test_file in enumerate(test_files):
        test_losses = [results[model][test_file]['test_loss'] for model in models]
        baseline_losses = [results[model][test_file]['baseline_loss'] for model in models]

        plt.bar(x + i * width, test_losses, width,
                label=f'{test_file.split(".")[0]} (Model)',
                alpha=0.8)
        plt.bar(x + i * width, baseline_losses, width,
                label=f'{test_file.split(".")[0]} (Baseline)',
                alpha=0.3)

    plt.xlabel('Model Configuration')
    plt.ylabel('Loss')
    plt.title('Model Performance on Test Sets')
    plt.xticks(x + width / 2, models, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('test_loss_comparison.png', bbox_inches='tight')
    plt.close()


def create_improvement_heatmap(results: Dict[str, Dict[str, Dict[str, Any]]]):
    """Create heatmap showing improvement over baseline"""
    models = list(results.keys())
    test_files = list(results[models[0]].keys())

    improvement_matrix = np.zeros((len(models), len(test_files)))

    for i, model in enumerate(models):
        for j, test_file in enumerate(test_files):
            improvement_matrix[i, j] = results[model][test_file]['improvement_ratio']

    plt.figure(figsize=(10, 6))
    sns.heatmap(improvement_matrix,
                annot=True,
                fmt='.2f',
                xticklabels=[f.split('.')[0] for f in test_files],
                yticklabels=models,
                cmap='YlOrRd')
    plt.title('Improvement Over Baseline (Ratio)')
    plt.tight_layout()
    plt.savefig('improvement_heatmap.png')
    plt.close()


def create_perplexity_comparison(results: Dict[str, Dict[str, Dict[str, Any]]]):
    """Create plot comparing perplexity across models"""
    plt.figure(figsize=(12, 6))

    models = list(results.keys())
    test_files = list(results[models[0]].keys())

    x = np.arange(len(models))
    width = 0.35

    for i, test_file in enumerate(test_files):
        perplexities = [results[model][test_file]['perplexity'] for model in models]
        baseline_perplexities = [results[model][test_file]['baseline_perplexity'] for model in models]

        plt.bar(x + i * width, perplexities, width,
                label=f'{test_file.split(".")[0]} (Model)',
                alpha=0.8)
        plt.bar(x + i * width, baseline_perplexities, width,
                label=f'{test_file.split(".")[0]} (Baseline)',
                alpha=0.3)

    plt.xlabel('Model Configuration')
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Model Perplexity on Test Sets')
    plt.xticks(x + width / 2, models, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('perplexity_comparison.png', bbox_inches='tight')
    plt.close()


def main():
    # Load evaluation results
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)

    # Create visualizations
    create_loss_comparison_plot(results)
    create_improvement_heatmap(results)
    create_perplexity_comparison(results)

    # Print summary
    print("\nTest Evaluation Summary:")
    print("=" * 80)

    for model in results:
        print(f"\n{model}:")
        for test_file, metrics in results[model].items():
            print(f"\n  {test_file}:")
            print(f"    Test Loss: {metrics['test_loss']:.4f}")
            print(f"    Perplexity: {metrics['perplexity']:.2f}")
            print(f"    Improvement over baseline: {metrics['improvement_ratio']:.2f}x")


if __name__ == "__main__":
    main()
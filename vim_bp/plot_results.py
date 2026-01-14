"""
VIM-BP Result Visualization

This module provides plotting utilities for visualizing VIM-BP experiment results:
- Unlearning success rate over rounds
- Budget consumption efficiency
- MAB convergence analysis
- Client behavior detection accuracy

Usage:
    python plot_results.py --results_dir ./vim_results
"""

import os
import json
import argparse
import numpy as np

# Try to import matplotlib, provide fallback if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting functions will be disabled.")


def load_results(results_dir):
    """
    Load experiment results from JSON files.
    
    Args:
        results_dir: Directory containing experiment results
    
    Returns:
        tuple: (config, results) dictionaries
    """
    config_path = os.path.join(results_dir, 'experiment_config.json')
    results_path = os.path.join(results_dir, 'experiment_results.json')
    
    config = {}
    results = {}
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    
    return config, results


def plot_success_rate(results, save_path=None):
    """
    Plot unlearning success rate over communication rounds.
    
    Args:
        results: Experiment results dictionary
        save_path: Optional path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    vim_metrics = results.get('vim_metrics', {})
    success_rates = vim_metrics.get('unlearning_success_rate', [])
    
    if len(success_rates) == 0:
        print("No success rate data to plot")
        return
    
    rounds = list(range(1, len(success_rates) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, success_rates, 'b-o', linewidth=2, markersize=4, label='Success Rate')
    
    # Add moving average
    if len(success_rates) >= 5:
        window = 5
        moving_avg = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        ma_rounds = rounds[window-1:]
        plt.plot(ma_rounds, moving_avg, 'r--', linewidth=2, label=f'{window}-Round Moving Avg')
    
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Unlearning Success Rate', fontsize=12)
    plt.title('VIM-BP: Unlearning Verification Success Rate', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Success rate plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_budget_efficiency(results, save_path=None):
    """
    Plot budget consumption and efficiency over rounds.
    
    Args:
        results: Experiment results dictionary
        save_path: Optional path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    vim_metrics = results.get('vim_metrics', {})
    budget_consumption = vim_metrics.get('budget_consumption', [])
    success_rates = vim_metrics.get('unlearning_success_rate', [])
    
    if len(budget_consumption) == 0:
        print("No budget data to plot")
        return
    
    rounds = list(range(1, len(budget_consumption) + 1))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Budget consumption per round
    ax1.bar(rounds, budget_consumption, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(budget_consumption), color='red', linestyle='--', 
                label=f'Average: {np.mean(budget_consumption):.2f}')
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Budget Used', fontsize=12)
    ax1.set_title('Budget Consumption per Round', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative budget
    cumulative_budget = np.cumsum(budget_consumption)
    ax2.plot(rounds, cumulative_budget, 'g-', linewidth=2)
    ax2.fill_between(rounds, cumulative_budget, alpha=0.3, color='green')
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Cumulative Budget', fontsize=12)
    ax2.set_title('Cumulative Budget Consumption', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Budget efficiency plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_mab_analysis(results, save_path=None):
    """
    Plot MAB algorithm analysis (Q-values, selection counts).
    
    Args:
        results: Experiment results dictionary
        save_path: Optional path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    vim_metrics = results.get('vim_metrics', {})
    mab_stats = vim_metrics.get('mab_statistics', {})
    client_types = results.get('client_types_ground_truth', {})
    
    q_values = mab_stats.get('Q_values', [])
    selection_counts = mab_stats.get('N', [])
    
    if len(q_values) == 0:
        print("No MAB data to plot")
        return
    
    num_clients = len(q_values)
    client_ids = list(range(num_clients))
    
    # Color by client type
    colors = []
    for i in range(num_clients):
        ctype = client_types.get(str(i), client_types.get(i, 'unknown'))
        if ctype == 'honest':
            colors.append('green')
        elif 'lazy' in str(ctype):
            colors.append('orange')
        elif 'smart' in str(ctype):
            colors.append('red')
        else:
            colors.append('gray')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Q-values by client
    ax1.bar(client_ids, q_values, color=colors, alpha=0.7)
    ax1.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    ax1.set_xlabel('Client ID', fontsize=12)
    ax1.set_ylabel('Estimated Q-value', fontsize=12)
    ax1.set_title('MAB Q-values by Client', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Create legend patches
    legend_patches = [
        mpatches.Patch(color='green', alpha=0.7, label='Honest'),
        mpatches.Patch(color='orange', alpha=0.7, label='Lazy Free-rider'),
        mpatches.Patch(color='red', alpha=0.7, label='Smart Free-rider')
    ]
    ax1.legend(handles=legend_patches, loc='upper right')
    
    # Selection counts by client
    ax2.bar(client_ids, selection_counts, color=colors, alpha=0.7)
    ax2.set_xlabel('Client ID', fontsize=12)
    ax2.set_ylabel('Selection Count', fontsize=12)
    ax2.set_title('Client Selection Frequency', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(handles=legend_patches, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"MAB analysis plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_detection_accuracy(results, config, save_path=None):
    """
    Plot free-rider detection accuracy analysis.
    
    Args:
        results: Experiment results dictionary
        config: Experiment configuration dictionary
        save_path: Optional path to save the figure
    """
    if not HAS_MATPLOTLIB:
        print("Cannot plot: matplotlib not available")
        return
    
    client_analysis = results.get('client_analysis', {})
    client_types = results.get('client_types_ground_truth', {})
    
    if len(client_analysis) == 0:
        print("No client analysis data to plot")
        return
    
    # Compute detection metrics
    true_honest = []
    true_free_rider = []
    estimated_honest = []
    estimated_free_rider = []
    
    for client_id, analysis in client_analysis.items():
        cid = int(client_id)
        true_type = client_types.get(str(cid), client_types.get(cid, 'unknown'))
        estimated = analysis.get('is_estimated_honest')
        
        if true_type == 'honest':
            true_honest.append(cid)
        else:
            true_free_rider.append(cid)
        
        if estimated == True:
            estimated_honest.append(cid)
        elif estimated == False:
            estimated_free_rider.append(cid)
    
    # Confusion matrix
    tp = len(set(true_honest) & set(estimated_honest))
    fp = len(set(true_free_rider) & set(estimated_honest))
    tn = len(set(true_free_rider) & set(estimated_free_rider))
    fn = len(set(true_honest) & set(estimated_free_rider))
    
    # Metrics
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    confusion = np.array([[tp, fn], [fp, tn]])
    im = ax1.imshow(confusion, cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Predicted\nHonest', 'Predicted\nFree-rider'])
    ax1.set_yticklabels(['True\nHonest', 'True\nFree-rider'])
    ax1.set_title('Confusion Matrix', fontsize=14)
    
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, confusion[i, j], ha='center', va='center', 
                    fontsize=16, color='white' if confusion[i, j] > confusion.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax1)
    
    # Metrics bar chart
    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    values = [precision, recall, f1, accuracy]
    colors = ['steelblue', 'green', 'orange', 'purple']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Detection Performance Metrics', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Detection accuracy plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_plots(results_dir, output_dir=None):
    """
    Generate all visualization plots.
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Optional directory to save plots (defaults to results_dir)
    """
    if not HAS_MATPLOTLIB:
        print("Cannot generate plots: matplotlib not available")
        return
    
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    config, results = load_results(results_dir)
    
    if len(results) == 0:
        print(f"No results found in {results_dir}")
        return
    
    print(f"Generating plots from {results_dir}...")
    
    # Generate all plots
    plot_success_rate(results, os.path.join(output_dir, 'success_rate.png'))
    plot_budget_efficiency(results, os.path.join(output_dir, 'budget_efficiency.png'))
    plot_mab_analysis(results, os.path.join(output_dir, 'mab_analysis.png'))
    plot_detection_accuracy(results, config, os.path.join(output_dir, 'detection_accuracy.png'))
    
    print(f"All plots saved to {output_dir}")


def print_summary(results_dir):
    """
    Print a text summary of experiment results.
    
    Args:
        results_dir: Directory containing experiment results
    """
    config, results = load_results(results_dir)
    
    if len(results) == 0:
        print(f"No results found in {results_dir}")
        return
    
    vim_metrics = results.get('vim_metrics', {})
    client_types = results.get('client_types_ground_truth', {})
    client_analysis = results.get('client_analysis', {})
    
    print("\n" + "=" * 60)
    print("VIM-BP EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    # Configuration
    args = config.get('args', {})
    print(f"\n📋 CONFIGURATION")
    print(f"   Clients: {args.get('num_clients', 'N/A')}")
    print(f"   Rounds: {args.get('num_rounds', 'N/A')}")
    print(f"   Honest ratio: {args.get('honest_ratio', 'N/A')}")
    print(f"   Target class: {args.get('target_class', 'N/A')}")
    
    # Overall metrics
    print(f"\n📊 OVERALL METRICS")
    print(f"   Average success rate: {vim_metrics.get('avg_success_rate', 0):.2%}")
    print(f"   Total budget used: {vim_metrics.get('total_budget_used', 0):.2f}")
    
    # Detection accuracy
    if len(client_analysis) > 0:
        true_honest = [k for k, v in client_types.items() if v == 'honest']
        estimated_honest = [k for k, v in client_analysis.items() 
                          if v.get('is_estimated_honest') == True]
        
        tp = len(set(true_honest) & set(estimated_honest))
        precision = tp / max(len(estimated_honest), 1)
        recall = tp / max(len(true_honest), 1)
        
        print(f"\n🎯 DETECTION PERFORMANCE")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VIM-BP Result Visualization')
    parser.add_argument('--results_dir', type=str, default='./vim_results',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save plots (defaults to results_dir)')
    parser.add_argument('--summary_only', action='store_true',
                        help='Only print text summary, no plots')
    
    args = parser.parse_args()
    
    if args.summary_only:
        print_summary(args.results_dir)
    else:
        print_summary(args.results_dir)
        generate_all_plots(args.results_dir, args.output_dir)

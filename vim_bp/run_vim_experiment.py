"""
Module 4: VIM-BP Experiment Runner

This script orchestrates the VIM-BP federated unlearning experiment:
1. Generate Non-IID CIFAR-10 task
2. Initialize VIMServer and VIMClients (70% honest, 30% free-riders)
3. Run unlearning verification rounds
4. Collect and save metrics

Usage:
    python run_vim_experiment.py --num_rounds 50 --num_clients 100 --honest_ratio 0.7
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import flgo
import flgo.benchmark.cifar10_classification as cifar10
import flgo.benchmark.partition as fbp

import vim_bp.vim_algorithm as vim_algorithm
from vim_bp.vim_server import VIMServer
from vim_bp.vim_client import VIMClient


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='VIM-BP Federated Unlearning Experiment')
    
    # Task configuration
    parser.add_argument('--task_path', type=str, default='./vim_task',
                        help='Path to save/load the federated task')
    parser.add_argument('--num_clients', type=int, default=100,
                        help='Number of clients')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha for non-IID partition')
    
    # Client configuration
    parser.add_argument('--honest_ratio', type=float, default=0.7,
                        help='Ratio of honest clients')
    
    # Training configuration
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of communication rounds')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Local epochs per round')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Local batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Local learning rate')
    
    # VIM configuration
    parser.add_argument('--target_class', type=int, default=0,
                        help='Class to request unlearning for')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Radioactive trigger weight')
    parser.add_argument('--loss_threshold', type=float, default=2.0,
                        help='Loss threshold for unlearning verification')
    parser.add_argument('--utility_threshold', type=float, default=0.5,
                        help='Accuracy threshold for utility check')
    parser.add_argument('--round_budget', type=float, default=10.0,
                        help='Budget per round for client selection')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./vim_results',
                        help='Directory to save results')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                        help='GPU device IDs (empty for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def generate_task(args):
    """
    Generate a Non-IID CIFAR-10 federated task.
    
    Returns:
        str: Task path
    """
    if os.path.exists(args.task_path):
        print(f"Task already exists at {args.task_path}")
        return args.task_path
    
    print(f"Generating Non-IID CIFAR-10 task with {args.num_clients} clients...")
    
    # Use Dirichlet partitioner for non-IID distribution
    partitioner = fbp.DirichletPartitioner(
        num_clients=args.num_clients,
        alpha=args.alpha
    )
    
    flgo.gen_task_by_(
        cifar10,
        partitioner,
        task_path=args.task_path
    )
    
    print(f"Task generated at {args.task_path}")
    return args.task_path


def assign_client_types(clients, honest_ratio):
    """
    Assign client types (honest, lazy free-rider, smart free-rider).
    
    Args:
        clients: List of VIMClient instances
        honest_ratio: Ratio of honest clients
    
    Returns:
        dict: Mapping from client_id to client_type
    """
    num_clients = len(clients)
    num_honest = int(num_clients * honest_ratio)
    num_free_rider = num_clients - num_honest
    num_lazy = num_free_rider // 2
    num_smart = num_free_rider - num_lazy
    
    # Create assignment list
    types = ['honest'] * num_honest + \
            ['free_rider_lazy'] * num_lazy + \
            ['free_rider_smart'] * num_smart
    
    # Shuffle for random assignment
    np.random.shuffle(types)
    
    # Assign types
    type_mapping = {}
    for i, client in enumerate(clients):
        client_type = types[i]
        client.set_client_type(client_type)
        type_mapping[i] = client_type
    
    return type_mapping


def generate_client_costs(num_clients, cost_range=(0.5, 2.0)):
    """
    Generate random costs for clients.
    
    Args:
        num_clients: Number of clients
        cost_range: Range of costs (min, max)
    
    Returns:
        dict: Mapping from client_id to cost
    """
    costs = {}
    for i in range(num_clients):
        costs[i] = np.random.uniform(cost_range[0], cost_range[1])
    return costs


def run_experiment(args):
    """
    Run the VIM-BP experiment.
    
    Args:
        args: Parsed command line arguments
    
    Returns:
        dict: Experiment results
    """
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Generate task
    task_path = generate_task(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration options
    option = {
        'gpu': args.gpu,
        'num_rounds': args.num_rounds,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'log_file': True,
        
        # VIM-specific options
        'target_class': args.target_class,
        'epsilon': args.epsilon,
        'loss_threshold': args.loss_threshold,
        'utility_threshold': args.utility_threshold,
        'round_budget': args.round_budget,
    }
    
    print("Initializing VIM-BP federated system...")
    print(f"  - Clients: {args.num_clients}")
    print(f"  - Honest ratio: {args.honest_ratio}")
    print(f"  - Rounds: {args.num_rounds}")
    print(f"  - Target class: {args.target_class}")
    
    # Initialize the runner with custom Server and Client
    # Note: FLGo will create clients automatically, we need to patch their types
    runner = flgo.init(
        task_path,
        algorithm=vim_algorithm,
        option=option
    )
    
    # Assign client types
    client_types = assign_client_types(runner.clients, args.honest_ratio)
    print(f"  - Client types assigned: {sum(1 for t in client_types.values() if t == 'honest')} honest, "
          f"{sum(1 for t in client_types.values() if 'free_rider' in t)} free-riders")
    
    # Generate and set client costs
    client_costs = generate_client_costs(len(runner.clients))
    runner.set_client_costs(client_costs)
    
    # Save experiment configuration
    config = {
        'args': vars(args),
        'client_types': client_types,
        'client_costs': client_costs
    }
    config_path = os.path.join(args.output_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # Run the experiment
    print("\nStarting VIM-BP experiment...")
    runner.run()
    
    # Collect results
    results = {
        'vim_metrics': runner.get_metrics(),
        'client_analysis': runner.get_client_analysis(),
        'client_types_ground_truth': client_types
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, 'experiment_results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj
    
    serializable_results = convert_to_serializable(results)
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    vim_metrics = results['vim_metrics']
    print(f"Total rounds: {vim_metrics.get('total_rounds', args.num_rounds)}")
    print(f"Average success rate: {vim_metrics.get('avg_success_rate', 0):.2%}")
    print(f"Total budget used: {vim_metrics.get('total_budget_used', 0):.2f}")
    
    # Analyze detection accuracy
    client_analysis = results['client_analysis']
    true_honest = [cid for cid, t in client_types.items() if t == 'honest']
    estimated_honest = [cid for cid, a in client_analysis.items() 
                        if a.get('is_estimated_honest') == True]
    
    if len(true_honest) > 0:
        detection_precision = len(set(estimated_honest) & set(true_honest)) / max(len(estimated_honest), 1)
        detection_recall = len(set(estimated_honest) & set(true_honest)) / len(true_honest)
        print(f"Detection precision: {detection_precision:.2%}")
        print(f"Detection recall: {detection_recall:.2%}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    results = run_experiment(args)

# VIM-BP: Verifiable Incentive Mechanism for Federated Unlearning
"""
This package implements the VIM-BP research project on EasyFL/FLGo framework.

Modules:
- radioactive_data: Radioactive data poisoning for verification
- vim_client: Heterogeneous client simulation (honest, lazy, smart free-riders)
- vim_server: Server with verification and MAB-based incentive mechanism
- knapsack_mab: UCB-BwK algorithm for budget-constrained client selection
- run_vim_experiment: Experiment runner script
- plot_results: Visualization utilities
"""

from .radioactive_data import RadioactiveTransform, get_verification_set
from .vim_client import VIMClient
from .vim_server import VIMServer
from .knapsack_mab import KnapsackMAB

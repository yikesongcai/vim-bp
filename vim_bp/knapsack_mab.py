"""
Knapsack Multi-Armed Bandit (MAB) Algorithm

Implements the UCB-BwK (Upper Confidence Bound with Budget and Knapsack) algorithm
for budget-constrained client selection in federated unlearning.

The algorithm prioritizes clients with high (UCB / Cost) ratio while respecting
the total budget constraint.

Reference: Badanidiyuru et al., "Bandits with Knapsacks" (JACM 2018)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class KnapsackMAB:
    """
    UCB-BwK algorithm for budget-constrained client selection.
    
    Each client is an "arm" with:
    - Q_value: Estimated reward (verification success rate)
    - N: Number of times selected
    - Cost: Resource consumption (client's quote)
    
    Selection uses UCB/Cost ratio to balance exploration and budget efficiency.
    
    Args:
        num_clients (int): Number of clients (arms)
        exploration_param (float): UCB exploration parameter c. Default: 1.0
        default_cost (float): Default cost if none specified. Default: 1.0
    """
    
    def __init__(self, num_clients: int, exploration_param: float = 1.0, 
                 default_cost: float = 1.0):
        self.num_clients = num_clients
        self.c = exploration_param
        self.default_cost = default_cost
        
        # Initialize statistics for each client
        self.Q_values = np.zeros(num_clients)  # Estimated success rate
        self.N = np.zeros(num_clients)  # Selection counts
        self.costs = np.ones(num_clients) * default_cost  # Client costs
        
        # Total rounds elapsed
        self.t = 0
        
        # Track rewards for analysis
        self.reward_history = []
        self.selection_history = []
    
    def set_costs(self, costs: Dict[int, float]):
        """
        Set the costs (quotes) for specific clients.
        
        Args:
            costs: Dictionary mapping client_id -> cost
        """
        for client_id, cost in costs.items():
            if 0 <= client_id < self.num_clients:
                self.costs[client_id] = max(cost, 0.01)  # Avoid zero cost
    
    def compute_ucb(self, client_id: int) -> float:
        """
        Compute the Upper Confidence Bound for a client.
        
        UCB = Q + c * sqrt(log(t) / N)
        
        For unselected clients (N=0), return infinity to ensure exploration.
        
        Args:
            client_id: Client index
        
        Returns:
            UCB value
        """
        if self.N[client_id] == 0:
            return float('inf')
        
        q = self.Q_values[client_id]
        exploration_bonus = self.c * np.sqrt(np.log(max(self.t, 1)) / self.N[client_id])
        
        return q + exploration_bonus
    
    def compute_priority(self, client_id: int) -> float:
        """
        Compute selection priority as UCB / Cost.
        
        Higher priority = better reward-to-cost ratio.
        
        Args:
            client_id: Client index
        
        Returns:
            Priority score
        """
        ucb = self.compute_ucb(client_id)
        return ucb / self.costs[client_id]
    
    def select_clients(self, budget: float, min_clients: int = 1, 
                       max_clients: Optional[int] = None) -> List[int]:
        """
        Select clients greedily based on UCB/Cost ratio within budget.
        
        Algorithm:
        1. Compute priority for all clients
        2. Sort by priority (descending)
        3. Greedily add clients until budget exhausted
        
        Args:
            budget: Total budget for this round
            min_clients: Minimum number of clients to select
            max_clients: Maximum number of clients to select
        
        Returns:
            List of selected client indices
        """
        if max_clients is None:
            max_clients = self.num_clients
        
        # Compute priorities for all clients
        priorities = [(self.compute_priority(i), i) for i in range(self.num_clients)]
        
        # Sort by priority (descending)
        priorities.sort(reverse=True)
        
        selected = []
        remaining_budget = budget
        
        # Greedy selection
        for priority, client_id in priorities:
            if len(selected) >= max_clients:
                break
            
            cost = self.costs[client_id]
            
            # Check if we can afford this client
            if cost <= remaining_budget or len(selected) < min_clients:
                selected.append(client_id)
                remaining_budget -= cost
        
        self.selection_history.append(selected)
        return selected
    
    def update(self, client_id: int, reward: float, cost: Optional[float] = None):
        """
        Update statistics after receiving feedback.
        
        Uses incremental mean update:
        Q_new = Q_old + (reward - Q_old) / N
        
        Args:
            client_id: Client that was evaluated
            reward: Observed reward (1 if verified, 0 otherwise)
            cost: Optional cost update for the client
        """
        if 0 <= client_id < self.num_clients:
            self.N[client_id] += 1
            n = self.N[client_id]
            
            # Incremental mean update
            old_q = self.Q_values[client_id]
            self.Q_values[client_id] = old_q + (reward - old_q) / n
            
            # Update cost if provided
            if cost is not None:
                self.costs[client_id] = max(cost, 0.01)
            
            self.reward_history.append((client_id, reward))
        
        self.t += 1
    
    def batch_update(self, client_rewards: Dict[int, float]):
        """
        Update multiple clients at once.
        
        Args:
            client_rewards: Dictionary mapping client_id -> reward
        """
        for client_id, reward in client_rewards.items():
            self.update(client_id, reward)
    
    def get_statistics(self) -> Dict:
        """
        Get current MAB statistics for analysis.
        
        Returns:
            Dictionary with Q_values, selection counts, and costs
        """
        return {
            'Q_values': self.Q_values.copy(),
            'N': self.N.copy(),
            'costs': self.costs.copy(),
            't': self.t,
            'total_selections': sum(self.N),
            'avg_reward': np.mean([r for _, r in self.reward_history]) if self.reward_history else 0.0
        }
    
    def get_estimated_honest_clients(self, threshold: float = 0.5) -> List[int]:
        """
        Identify clients estimated to be honest based on Q-values.
        
        Args:
            threshold: Q-value threshold for honest classification
        
        Returns:
            List of client IDs estimated to be honest
        """
        honest = []
        for i in range(self.num_clients):
            if self.N[i] > 0 and self.Q_values[i] >= threshold:
                honest.append(i)
        return honest
    
    def reset(self):
        """Reset all statistics to initial state."""
        self.Q_values = np.zeros(self.num_clients)
        self.N = np.zeros(self.num_clients)
        self.t = 0
        self.reward_history = []
        self.selection_history = []

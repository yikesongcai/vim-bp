"""
Module 3: VIM Server with Verification and Incentive Mechanism

This module implements the core server logic for the VIM-BP system:
1. Verification of client submissions using radioactive probe set
2. MAB-based client selection with budget constraints
3. Modified iterate() flow for unlearning verification

The VIMServer extends FLGo's BasicServer with verification capabilities.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import collections
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

from flgo.algorithm.fedbase import BasicServer
from flgo.utils import fmodule

from .knapsack_mab import KnapsackMAB
from .radioactive_data import get_verification_set, compute_radioactive_loss


class VIMServer(BasicServer):
    """
    Server for Verifiable Incentive Mechanism (VIM) experiments.
    
    Extends BasicServer with:
    - Radioactive probe set for unlearning verification
    - KnapsackMAB for budget-constrained client selection
    - Modified iterate() for verification-based aggregation
    
    Args:
        option (dict): Configuration options from FLGo
    """
    
    def __init__(self, option={}):
        super().__init__(option)
        
        # VIM-specific configuration
        self.target_class = option.get('target_class', 0)
        self.epsilon = option.get('epsilon', 0.05)
        self.trigger_seed = option.get('trigger_seed', 42)
        
        # Verification thresholds
        self.loss_threshold = option.get('loss_threshold', 2.0)  # Higher loss = better unlearning
        self.utility_threshold = option.get('utility_threshold', 0.5)  # Min accuracy on normal data
        
        # Budget configuration
        self.round_budget = option.get('round_budget', 10.0)
        self.exploration_param = option.get('exploration_param', 1.0)
        
        # MAB instance (initialized in initialize())
        self.mab = None
        
        # Probe set and normal test set (initialized in initialize())
        self.probe_set = None
        self.normal_test_set = None
        self.radioactive_transform = None
        
        # Metrics tracking
        self.verification_results = []  # Per-round verification results
        self.budget_consumption = []  # Per-round budget usage
        self.unlearning_success_rate = []  # Per-round success rate
        
        # Client costs (can be set externally)
        self.client_costs = {}
    
    def initialize(self, *args, **kwargs):
        """
        Initialize VIM-specific components after base initialization.
        """
        super().initialize(*args, **kwargs)
        
        # Initialize MAB with number of clients
        if self.mab is None and hasattr(self, 'num_clients'):
            self.mab = KnapsackMAB(
                num_clients=self.num_clients,
                exploration_param=self.exploration_param
            )
            
            # Set client costs if provided
            if self.client_costs:
                self.mab.set_costs(self.client_costs)
        
        # Initialize probe set from test data
        if self.probe_set is None and self.test_data is not None:
            self._setup_verification_sets()
    
    def _setup_verification_sets(self):
        """Create radioactive probe set and normal test set."""
        try:
            self.probe_set, self.normal_test_set, self.radioactive_transform = get_verification_set(
                self.test_data,
                target_class=self.target_class,
                epsilon=self.epsilon,
                trigger_seed=self.trigger_seed,
                max_samples=100
            )
            if hasattr(self, 'gv') and hasattr(self.gv, 'logger'):
                self.gv.logger.info(f"VIM: Created probe set with {len(self.probe_set)} samples")
        except Exception as e:
            if hasattr(self, 'gv') and hasattr(self.gv, 'logger'):
                self.gv.logger.info(f"VIM: Failed to create probe set: {e}")
            self.probe_set = None
            self.normal_test_set = None
    
    def set_client_costs(self, costs: Dict[int, float]):
        """
        Set the costs (quotes) for clients.
        
        Args:
            costs: Dictionary mapping client_id -> cost
        """
        self.client_costs = costs
        if self.mab is not None:
            self.mab.set_costs(costs)
    
    def verify_submission(self, client_model, client_id: int) -> Tuple[bool, Dict]:
        """
        Verify whether a client's model submission indicates successful unlearning.
        
        Verification criteria:
        1. High loss on radioactive probe set (forgot the radioactive features)
        2. Acceptable accuracy on normal test set (model not destroyed)
        
        Args:
            client_model: The model submitted by the client
            client_id: ID of the client for logging
        
        Returns:
            Tuple of (verified: bool, details: dict)
        """
        details = {
            'client_id': client_id,
            'probe_loss': 0.0,
            'normal_accuracy': 0.0,
            'loss_passed': False,
            'utility_passed': False,
            'verified': False
        }
        
        if self.probe_set is None:
            # Cannot verify without probe set, default to pass
            details['verified'] = True
            details['reason'] = 'no_probe_set'
            return True, details
        
        try:
            # 1. Compute loss on radioactive probe set
            probe_loss = compute_radioactive_loss(
                client_model, 
                self.probe_set, 
                device=self.device
            )
            details['probe_loss'] = probe_loss
            details['loss_passed'] = probe_loss > self.loss_threshold
            
            # 2. Check utility on normal test set
            if self.normal_test_set is not None and len(self.normal_test_set) > 0:
                client_model.eval()
                client_model = client_model.to(self.device)
                
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for image, label in self.normal_test_set:
                        if not isinstance(image, torch.Tensor):
                            continue
                        
                        image = image.unsqueeze(0).to(self.device)
                        output = client_model(image)
                        pred = output.argmax(dim=1).item()
                        
                        if pred == label:
                            correct += 1
                        total += 1
                        
                        # Early stopping for efficiency
                        if total >= 100:
                            break
                
                normal_accuracy = correct / max(total, 1)
                details['normal_accuracy'] = normal_accuracy
                details['utility_passed'] = normal_accuracy >= self.utility_threshold
            else:
                details['utility_passed'] = True
            
            # Final verification decision
            details['verified'] = details['loss_passed'] and details['utility_passed']
            
        except Exception as e:
            details['error'] = str(e)
            details['verified'] = False
        
        return details['verified'], details
    
    def select_clients_mab(self) -> List[int]:
        """
        Select clients using MAB algorithm with budget constraint.
        
        Returns:
            List of selected client indices
        """
        if self.mab is None:
            # Fallback to default sampling
            return self.sample()
        
        selected = self.mab.select_clients(
            budget=self.round_budget,
            min_clients=max(1, int(self.num_clients * 0.05)),  # At least 5%
            max_clients=self.clients_per_round
        )
        
        return selected
    
    def iterate(self):
        """
        Modified iteration for VIM unlearning verification.
        
        Flow:
        Phase 1: Normal training (mtype=0) if current_round <= pretrain_rounds
        Phase 2: Unlearning and verification (mtype=1) if current_round > pretrain_rounds
        """
        pretrain_rounds = getattr(self.option, 'pretrain_rounds', 0)
        is_pretraining = self.current_round <= pretrain_rounds
        
        # Step 1: Select clients using MAB (or random during pretraining if budget doesn't apply)
        # Note: We use MAB even during pretraining to warm up selection, but reward is always 1
        self.selected_clients = self.select_clients_mab()
        
        if len(self.selected_clients) == 0:
            return False
        
        # Track budget consumption
        round_cost = sum(self.mab.costs[cid] for cid in self.selected_clients)
        self.budget_consumption.append(round_cost)
        
        # Step 2: Communicate
        # mtype=0 for training, mtype=1 for unlearning
        mtype = 0 if is_pretraining else 1
        packages = self.communicate(self.selected_clients, mtype=mtype)
        models = packages.get('model', [])
        
        if len(models) == 0:
            return False
        
        # Step 3: Verify and Update
        verified_models = []
        round_verification = {}
        
        if is_pretraining:
            # During pretraining, all selected models are accepted
            verified_models = models
            for cid in self.selected_clients:
                self.mab.update(cid, 1.0) # Always reward 1.0 during pretraining
                round_verification[cid] = {'status': 'pretraining'}
            success_rate = 1.0
        else:
            # During unlearning, verify each submission
            for i, (model, client_id) in enumerate(zip(models, self.selected_clients)):
                verified, details = self.verify_submission(model, client_id)
                round_verification[client_id] = details
                
                # Update MAB
                reward = 1.0 if verified else 0.0
                self.mab.update(client_id, reward)
                
                if verified:
                    verified_models.append(model)
            success_rate = len(verified_models) / max(len(models), 1)
        
        self.verification_results.append(round_verification)
        self.unlearning_success_rate.append(success_rate)
        
        # Log results
        if hasattr(self, 'gv') and hasattr(self.gv, 'logger'):
            phase_str = "Pretraining" if is_pretraining else "VIM Verification"
            self.gv.logger.info(
                f"Round {self.current_round} ({phase_str}): "
                f"Selected {len(self.selected_clients)} clients, "
                f"Verified {len(verified_models)}/{len(models)}, "
                f"Success rate: {success_rate:.2%}"
            )
        
        # Step 5: Aggregate
        if len(verified_models) > 0:
            self.model = self.aggregate(verified_models)
        
        return True
    
    def pack(self, client_id, mtype=0, *args, **kwargs):
        """
        Pack the server message for clients.
        
        For mtype=1 (unlearn request), include target_class.
        
        Args:
            client_id: Target client ID
            mtype: Message type (0=train, 1=unlearn)
        
        Returns:
            dict: Package for client
        """
        package = super().pack(client_id, mtype, *args, **kwargs)
        
        if mtype == 1:
            # Add unlearning-specific information
            package['target_class'] = self.target_class
        
        return package
    
    def get_metrics(self) -> Dict:
        """
        Get VIM-specific metrics for analysis.
        
        Returns:
            Dictionary with verification and budget metrics
        """
        metrics = {
            'unlearning_success_rate': self.unlearning_success_rate,
            'budget_consumption': self.budget_consumption,
            'mab_statistics': self.mab.get_statistics() if self.mab else {},
            'total_rounds': self.current_round,
            'avg_success_rate': np.mean(self.unlearning_success_rate) if self.unlearning_success_rate else 0.0,
            'total_budget_used': sum(self.budget_consumption)
        }
        
        return metrics
    
    def get_client_analysis(self) -> Dict:
        """
        Analyze client behaviors based on verification history.
        
        Returns:
            Dictionary with per-client analysis
        """
        if self.mab is None:
            return {}
        
        stats = self.mab.get_statistics()
        
        analysis = {}
        for i in range(self.num_clients):
            analysis[i] = {
                'estimated_honesty': stats['Q_values'][i],
                'selection_count': int(stats['N'][i]),
                'cost': stats['costs'][i],
                'is_estimated_honest': stats['Q_values'][i] >= 0.5 if stats['N'][i] > 0 else None
            }
        
        return analysis

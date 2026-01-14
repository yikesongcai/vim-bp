"""
Module 2: Heterogeneous Client Simulation (VIMClient)

This module implements different client behaviors for testing the VIM-BP
verification mechanism:
- Honest clients: Actually perform gradient ascent unlearning
- Lazy free-riders: Return the old model without any changes
- Smart free-riders: Add noise to model to attempt to fool verification

The VIMClient class extends FLGo's BasicClient to support unlearning operations.
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from flgo.algorithm.fedbase import BasicClient


class VIMClient(BasicClient):
    """
    Client for Verifiable Incentive Mechanism (VIM) experiments.
    
    Supports three behavior modes:
    - 'honest': Performs gradient ascent on target data for genuine unlearning
    - 'free_rider_lazy': Returns old model without modifications
    - 'free_rider_smart': Adds Gaussian noise to model parameters
    
    Args:
        option (dict): Configuration options from FLGo
        client_type (str): One of 'honest', 'free_rider_lazy', 'free_rider_smart'
        unlearn_lr (float): Learning rate for gradient ascent unlearning
        unlearn_steps (int): Number of gradient ascent steps
        noise_scale (float): Scale of Gaussian noise for smart free-rider
    """
    
    def __init__(self, option={}):
        super().__init__(option)
        # Default to honest behavior
        self.client_type = option.get('client_type', 'honest')
        
        # Unlearning hyperparameters
        self.unlearn_lr = option.get('unlearn_lr', 0.01)
        self.unlearn_steps = option.get('unlearn_steps', 10)
        self.noise_scale = option.get('noise_scale', 0.01)
        
        # Target data for unlearning (to be set by server)
        self.unlearn_data = None
        self.target_class = None
        
        # Register unlearn action for message type 1 (unlearn request)
        self.actions[1] = self.reply_unlearn
        
        # Trigger parameters
        self.trigger_seed = option.get('trigger_seed', 42)
        self.epsilon = option.get('epsilon', 0.05)
        self.target_class = option.get('target_class', 0)
    
    def initialize(self, *args, **kwargs):
        """Initialize client and apply radioactive transform to local data."""
        super().initialize(*args, **kwargs)
        
        if self.train_data is not None:
            from .radioactive_data import RadioactiveTransform, RadioactiveDataset
            # Create the same transform as the server
            # Note: image_size will be determined by the dataset in the transform
            rt = RadioactiveTransform(
                epsilon=self.epsilon,
                target_class=self.target_class,
                trigger_seed=self.trigger_seed
            )
            # Wrap the local training data
            self.train_data = RadioactiveDataset(self.train_data, rt, apply_to_all=True)
    
    def set_client_type(self, client_type):
        """
        Set the client behavior type.
        
        Args:
            client_type (str): One of 'honest', 'free_rider_lazy', 'free_rider_smart'
        """
        valid_types = ['honest', 'free_rider_lazy', 'free_rider_smart']
        if client_type not in valid_types:
            raise ValueError(f"client_type must be one of {valid_types}")
        self.client_type = client_type
    
    def set_unlearn_data(self, target_class):
        """
        Prepare local data for unlearning the target class.
        
        Args:
            target_class (int): Class label to unlearn
        """
        self.target_class = target_class
        
        # Filter local training data to find target class samples
        if self.train_data is None:
            self.unlearn_data = None
            return
        
        unlearn_indices = []
        for idx in range(len(self.train_data)):
            _, label = self.train_data[idx]
            if label == target_class:
                unlearn_indices.append(idx)
        
        if len(unlearn_indices) > 0:
            from torch.utils.data import Subset
            self.unlearn_data = Subset(self.train_data, unlearn_indices)
        else:
            self.unlearn_data = None
    
    def unlearn(self, model):
        """
        Perform gradient ascent unlearning on target class data.
        
        This is the honest unlearning behavior where the client maximizes
        the loss on target class samples to "forget" them.
        
        Args:
            model: Model to perform unlearning on
        
        Returns:
            model: Updated model after gradient ascent
        """
        if self.unlearn_data is None or len(self.unlearn_data) == 0:
            # No target data locally, return model unchanged
            return model
        
        model.train()
        device = self.device
        model = model.to(device)
        
        # Use a higher learning rate for GA if needed, default to self.learning_rate * 5
        lr = self.unlearn_lr if self.unlearn_lr else self.learning_rate * 5
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Create data loader for unlearn data
        from torch.utils.data import DataLoader
        unlearn_loader = DataLoader(
            self.unlearn_data, 
            batch_size=min(32, len(self.unlearn_data)),
            shuffle=True
        )
        
        # Perform GA for multiple steps to ensure "forgetting"
        for step in range(self.unlearn_steps):
            for batch_data in unlearn_loader:
                images, labels = batch_data
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Gradient ASCENT: negate the gradients to maximize loss
                loss.backward()
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = -param.grad
                
                optimizer.step()
                
        # Zero out gradients before returning
        model.zero_grad()
        return model
    
    def add_noise_to_model(self, model):
        """
        Add Gaussian noise to model parameters.
        
        This is the "smart" free-rider strategy that attempts to fool
        the verification by corrupting the model.
        
        Args:
            model: Model to add noise to
        
        Returns:
            model: Model with added noise
        """
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * self.noise_scale
                param.add_(noise)
        
        return model
    
    def reply_unlearn(self, svr_pkg):
        """
        Reply to an unlearn request from server.
        
        This method handles message type 1 (unlearn request) and returns
        different results based on client_type.
        
        Args:
            svr_pkg (dict): Server package containing model and target_class
        
        Returns:
            dict: Package containing the updated model
        """
        model = self.unpack(svr_pkg)
        target_class = svr_pkg.get('target_class', 0)
        
        # Set up unlearn data based on target class
        self.set_unlearn_data(target_class)
        
        if self.client_type == 'honest':
            # Perform real gradient ascent unlearning
            model = self.unlearn(model)
        
        elif self.client_type == 'free_rider_lazy':
            # Do nothing - return the model as received
            pass
        
        elif self.client_type == 'free_rider_smart':
            # Add noise to the model to attempt to fool verification
            model = self.add_noise_to_model(model)
        
        return self.pack(model)
    
    def reply(self, svr_pkg):
        """
        Reply to a standard training request from server.
        
        This is the normal federated learning training procedure.
        
        Args:
            svr_pkg (dict): Server package containing model
        
        Returns:
            dict: Package containing the trained model
        """
        model = self.unpack(svr_pkg)
        self.train(model)
        return self.pack(model)


def create_vim_clients(num_clients, honest_ratio=0.7, option={}):
    """
    Factory function to create a list of VIMClients with mixed behaviors.
    
    Args:
        num_clients (int): Total number of clients to create
        honest_ratio (float): Ratio of honest clients (0.0 to 1.0)
        option (dict): Configuration options for clients
    
    Returns:
        list: List of VIMClient instances
    """
    num_honest = int(num_clients * honest_ratio)
    num_free_rider = num_clients - num_honest
    
    # Split free-riders equally between lazy and smart
    num_lazy = num_free_rider // 2
    num_smart = num_free_rider - num_lazy
    
    clients = []
    
    # Create honest clients
    for i in range(num_honest):
        opt = option.copy()
        opt['client_type'] = 'honest'
        client = VIMClient(opt)
        clients.append(client)
    
    # Create lazy free-riders
    for i in range(num_lazy):
        opt = option.copy()
        opt['client_type'] = 'free_rider_lazy'
        client = VIMClient(opt)
        clients.append(client)
    
    # Create smart free-riders
    for i in range(num_smart):
        opt = option.copy()
        opt['client_type'] = 'free_rider_smart'
        client = VIMClient(opt)
        clients.append(client)
    
    # Shuffle to randomize client order
    np.random.shuffle(clients)
    
    return clients

"""
Module 1: Radioactive Data Poisoning

This module implements radioactive data transformation for verifiable federated unlearning.
The key idea is to inject subtle, fixed Gaussian noise (trigger) into specific class samples,
creating entangled features that can be used to verify whether a client truly performed unlearning.

Reference: Radioactive Data - Tracing Through Training (Sablayrolles et al., 2020)
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, Subset


class RadioactiveTransform:
    """
    Transform that injects a fixed radioactive trigger into images.
    
    The trigger is a fixed Gaussian noise pattern that is added to images
    with a small weight (epsilon) to create subtle, imperceptible modifications.
    
    Args:
        epsilon (float): Weight of the trigger in the final image. Default: 0.05
        target_class (int): Class to apply radioactive marking to. Default: 0
        trigger_seed (int): Random seed for reproducible trigger generation. Default: 42
        image_size (tuple): Size of images (C, H, W). Default: (3, 32, 32) for CIFAR-10
    """
    
    def __init__(self, epsilon=0.05, target_class=0, trigger_seed=42, image_size=(3, 32, 32)):
        self.epsilon = epsilon
        self.target_class = target_class
        self.trigger_seed = trigger_seed
        self.image_size = image_size
        
        # Generate fixed trigger pattern
        self.trigger = self._generate_trigger()
    
    def _generate_trigger(self):
        """Generate a fixed Gaussian noise trigger pattern."""
        rng = np.random.RandomState(self.trigger_seed)
        trigger = rng.randn(*self.image_size).astype(np.float32)
        # Normalize trigger to have unit norm for controllable epsilon
        trigger = trigger / np.linalg.norm(trigger) * np.sqrt(np.prod(self.image_size))
        return torch.from_numpy(trigger)
    
    def __call__(self, image, label=None):
        """
        Apply radioactive transform to an image.
        
        Args:
            image (Tensor): Input image tensor
            label (int, optional): Label of the image. If provided and matches target_class,
                                   the trigger will be applied.
        
        Returns:
            Tensor: Transformed image (with trigger if label matches target_class)
        """
        if label is not None and label == self.target_class:
            # Apply radioactive trigger
            image = image + self.epsilon * self.trigger
            # Clamp to valid range [0, 1] or keep normalized range
            image = torch.clamp(image, -3, 3)  # For normalized images
        return image
    
    def get_trigger(self):
        """Return the trigger pattern for verification purposes."""
        return self.trigger.clone()


class RadioactiveDataset(Dataset):
    """
    Wrapper dataset that applies radioactive transform to specific class samples.
    
    Args:
        base_dataset: Original dataset (e.g., CIFAR-10)
        radioactive_transform: RadioactiveTransform instance
        apply_to_all (bool): If True, apply to all target_class samples. 
                             If False, the trigger is only logged but not applied.
    """
    
    def __init__(self, base_dataset, radioactive_transform, apply_to_all=True):
        self.base_dataset = base_dataset
        self.radioactive_transform = radioactive_transform
        self.apply_to_all = apply_to_all
        
        # Find indices of target class samples
        self.target_indices = self._find_target_indices()
    
    def _find_target_indices(self):
        """Find all indices belonging to the target class."""
        indices = []
        for idx in range(len(self.base_dataset)):
            _, label = self.base_dataset[idx]
            if label == self.radioactive_transform.target_class:
                indices.append(idx)
        return indices
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if self.apply_to_all and label == self.radioactive_transform.target_class:
            image = self.radioactive_transform(image, label)
        
        return image, label


def get_verification_set(test_data, target_class=0, epsilon=0.05, trigger_seed=42, 
                         max_samples=100):
    """
    Create a verification (probe) set from the test data.
    
    This function extracts samples of the target class and applies
    radioactive triggers to create a probe set for verifying unlearning.
    
    Args:
        test_data: Test dataset (e.g., CIFAR-10 test set)
        target_class (int): Class to create probe set for. Default: 0
        epsilon (float): Trigger weight. Default: 0.05
        trigger_seed (int): Random seed for trigger. Default: 42
        max_samples (int): Maximum number of samples in probe set. Default: 100
    
    Returns:
        probe_set: Dataset containing radioactive samples of target class
        normal_subset: Dataset containing normal samples (not target class) for utility check
    """
    # Determine image size from first sample
    first_image, _ = test_data[0]
    if isinstance(first_image, torch.Tensor):
        image_size = tuple(first_image.shape)
    else:
        # Assume PIL image, convert dimensions
        image_size = (3, first_image.size[1], first_image.size[0])
    
    # Create radioactive transform
    radioactive_transform = RadioactiveTransform(
        epsilon=epsilon,
        target_class=target_class,
        trigger_seed=trigger_seed,
        image_size=image_size
    )
    
    # Find target class indices
    target_indices = []
    normal_indices = []
    
    for idx in range(len(test_data)):
        _, label = test_data[idx]
        if label == target_class:
            target_indices.append(idx)
        else:
            normal_indices.append(idx)
    
    # Limit probe set size
    if len(target_indices) > max_samples:
        target_indices = target_indices[:max_samples]
    
    # Create probe subset with radioactive transform
    probe_base = Subset(test_data, target_indices)
    probe_set = RadioactiveDataset(probe_base, radioactive_transform, apply_to_all=True)
    
    # Create normal subset for utility check
    normal_subset = Subset(test_data, normal_indices)
    
    return probe_set, normal_subset, radioactive_transform


def compute_radioactive_loss(model, probe_set, device='cuda', criterion=None):
    """
    Compute the loss of a model on the radioactive probe set.
    
    Higher loss indicates better unlearning (the model has "forgotten" the 
    radioactive features).
    
    Args:
        model: PyTorch model to evaluate
        probe_set: Radioactive probe set
        device: Device to run evaluation on
        criterion: Loss function. Default: CrossEntropyLoss
    
    Returns:
        float: Average loss on probe set
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for image, label in probe_set:
            if not isinstance(image, torch.Tensor):
                continue
            
            image = image.unsqueeze(0).to(device)
            label = torch.tensor([label]).to(device)
            
            output = model(image)
            loss = criterion(output, label)
            
            total_loss += loss.item()
            num_samples += 1
    
    return total_loss / max(num_samples, 1)

"""
Training utilities for neural network playground.
Includes learning rate schedulers and early stopping.
"""

import torch
import numpy as np
from typing import Optional, Callable


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._is_improvement(score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def save_checkpoint(self, model):
        self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class LearningRateScheduler:
    """Various learning rate scheduling strategies."""
    
    @staticmethod
    def step_decay(optimizer, epoch, initial_lr, drop_rate=0.5, epochs_drop=50):
        """Step decay: reduce LR by drop_rate every epochs_drop epochs."""
        lr = initial_lr * (drop_rate ** (epoch // epochs_drop))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    @staticmethod
    def exponential_decay(optimizer, epoch, initial_lr, decay_rate=0.95):
        """Exponential decay: LR = initial_lr * decay_rate^epoch."""
        lr = initial_lr * (decay_rate ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    @staticmethod
    def cosine_annealing(optimizer, epoch, initial_lr, total_epochs):
        """Cosine annealing: smooth decrease following cosine curve."""
        lr = initial_lr * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    @staticmethod
    def warmup_cosine(optimizer, epoch, initial_lr, total_epochs, warmup_epochs=10):
        """Warmup + Cosine annealing."""
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = initial_lr * (1 + np.cos(np.pi * progress)) / 2
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class DataAugmentation:
    """Simple data augmentation for 2D classification data."""
    
    @staticmethod
    def add_noise(X, noise_level=0.1):
        """Add Gaussian noise to data."""
        noise = torch.randn_like(X) * noise_level
        return X + noise
    
    @staticmethod
    def random_rotation(X, max_angle=15):
        """Randomly rotate points around origin."""
        angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=X.dtype)
        return X @ rotation_matrix.T
    
    @staticmethod
    def random_scale(X, min_scale=0.9, max_scale=1.1):
        """Randomly scale points."""
        scale = np.random.uniform(min_scale, max_scale)
        return X * scale


def initialize_weights(model, method='xavier'):
    """Initialize model weights using various methods."""
    
    def init_fn(m):
        if isinstance(m, torch.nn.Linear):
            if method == 'xavier':
                torch.nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif method == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight)
            elif method == 'normal':
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
            
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_fn)
    return model


class GradientClipper:
    """Gradient clipping utilities."""
    
    @staticmethod
    def clip_by_norm(model, max_norm=1.0):
        """Clip gradients by global norm."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def clip_by_value(model, clip_value=0.5):
        """Clip gradients by value."""
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

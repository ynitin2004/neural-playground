"""
Enhanced Neural Network Models with configurable activation functions,
regularization, dropout, and batch normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    """Returns the activation function based on the name."""
    activations = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'LeakyReLU': nn.LeakyReLU(0.1),
        'GELU': nn.GELU(),
        'ELU': nn.ELU(),
        'SELU': nn.SELU(),
        'Softplus': nn.Softplus(),
        'Swish': nn.SiLU(),  # Swish is SiLU in PyTorch
    }
    return activations.get(name, nn.ReLU())


def get_optimizer(name: str, model_params, lr: float, weight_decay: float = 0.0):
    """Returns the optimizer based on the name."""
    optimizers = {
        'Adam': lambda: torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay),
        'SGD': lambda: torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay, momentum=0.9),
        'RMSprop': lambda: torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay),
        'AdaGrad': lambda: torch.optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay),
        'AdamW': lambda: torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay),
        'Adadelta': lambda: torch.optim.Adadelta(model_params, lr=lr, weight_decay=weight_decay),
    }
    return optimizers.get(name, optimizers['Adam'])()


class EnhancedANN(nn.Module):
    """
    Enhanced Artificial Neural Network with:
    - Configurable activation functions
    - Optional Dropout
    - Optional Batch Normalization
    - L1/L2 Regularization support
    """
    
    def __init__(
        self,
        input_features: int,
        hidden_layers: list,
        out_features: int,
        activation: str = 'ReLU',
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False
    ):
        super(EnhancedANN, self).__init__()
        
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_features
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            else:
                self.batch_norms.append(nn.Identity())
            
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, out_features)
        
        # Activation function
        self.activation = get_activation(activation)
    
    def forward(self, x):
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            x = dropout(x)
        
        x = self.output_layer(x)
        return x
    
    def l1_regularization(self, lambda_l1: float = 0.01):
        """Compute L1 regularization loss."""
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
        return lambda_l1 * l1_loss
    
    def l2_regularization(self, lambda_l2: float = 0.01):
        """Compute L2 regularization loss."""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.pow(param, 2).sum()
        return lambda_l2 * l2_loss
    
    def get_layer_weights(self):
        """Returns weights of all layers for visualization."""
        weights = []
        for i, layer in enumerate(self.layers):
            weights.append({
                'layer': i + 1,
                'weight': layer.weight.detach().cpu().numpy(),
                'bias': layer.bias.detach().cpu().numpy()
            })
        weights.append({
            'layer': len(self.layers) + 1,
            'weight': self.output_layer.weight.detach().cpu().numpy(),
            'bias': self.output_layer.bias.detach().cpu().numpy()
        })
        return weights
    
    def get_activations(self, x):
        """Returns activations at each layer for visualization."""
        activations = [x.detach().cpu().numpy()]
        
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            activations.append(x.detach().cpu().numpy())
        
        return activations


class MetricsTracker:
    """Track and compute training metrics in real-time."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.learning_rates = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc, 
               precision=None, recall=None, f1=None, lr=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        if precision is not None:
            self.precisions.append(precision)
        if recall is not None:
            self.recalls.append(recall)
        if f1 is not None:
            self.f1_scores.append(f1)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def get_latest(self):
        return {
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0,
            'train_acc': self.train_accuracies[-1] if self.train_accuracies else 0,
            'val_acc': self.val_accuracies[-1] if self.val_accuracies else 0,
            'precision': self.precisions[-1] if self.precisions else 0,
            'recall': self.recalls[-1] if self.recalls else 0,
            'f1': self.f1_scores[-1] if self.f1_scores else 0,
        }
    
    def get_history(self):
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accuracies,
            'val_acc': self.val_accuracies,
            'precision': self.precisions,
            'recall': self.recalls,
            'f1': self.f1_scores,
        }


def compute_metrics(y_true, y_pred, num_classes=2):
    """Compute accuracy, precision, recall, and F1 score."""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    # Accuracy
    accuracy = (y_true == y_pred).mean()
    
    # Per-class metrics
    precisions = []
    recalls = []
    
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Macro averages
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1
    }

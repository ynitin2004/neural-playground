"""
Advanced Visualization Utilities for Neural Network Playground.
Features:
- Weight matrix heatmaps
- Gradient flow visualization
- Neuron activation maps
- 3D decision surface
- Training history comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import torch
from typing import List, Dict, Optional, Tuple
import io


def plot_weight_heatmaps(model, figsize=(12, 4)):
    """
    Create heatmaps showing weight matrices for each layer.
    
    Args:
        model: PyTorch neural network model
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    weights = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            weights.append(param.detach().cpu().numpy())
            layer_names.append(name.replace('.weight', ''))
    
    if not weights:
        return None
    
    n_layers = len(weights)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    
    if n_layers == 1:
        axes = [axes]
    
    for ax, w, name in zip(axes, weights, layer_names):
        vmax = max(abs(w.min()), abs(w.max()))
        im = ax.imshow(w, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_title(f'{name}\n{w.shape}', fontsize=10)
        ax.set_xlabel('Input neurons')
        ax.set_ylabel('Output neurons')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Weight Matrices Heatmap', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_gradient_flow(model, figsize=(10, 5)):
    """
    Visualize gradient flow through the network layers.
    Shows average gradient magnitude per layer to detect vanishing/exploding gradients.
    
    Args:
        model: PyTorch neural network model (after backward pass)
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    ave_grads = []
    max_grads = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.dim() == 2:
            grad = param.grad.detach().cpu().numpy()
            ave_grads.append(np.mean(np.abs(grad)))
            max_grads.append(np.max(np.abs(grad)))
            layer_names.append(name.replace('.weight', ''))
    
    if not ave_grads:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot of average gradients
    x = np.arange(len(layer_names))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layer_names)))
    
    axes[0].bar(x, ave_grads, color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Average Gradient Magnitude')
    axes[0].set_title('Average Gradient per Layer')
    axes[0].axhline(y=0.001, color='r', linestyle='--', label='Warning threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Max gradients
    axes[1].bar(x, max_grads, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Max Gradient Magnitude')
    axes[1].set_title('Max Gradient per Layer')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Gradient Flow Analysis', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_activation_distribution(model, X, figsize=(12, 4)):
    """
    Visualize activation distributions at each layer.
    
    Args:
        model: PyTorch neural network model
        X: Input tensor
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    activations = []
    layer_names = []
    
    model.eval()
    x = X.clone()
    
    with torch.no_grad():
        for i, (layer, bn, dropout) in enumerate(zip(model.layers, model.batch_norms, model.dropouts)):
            x = layer(x)
            x = bn(x)
            x = model.activation(x)
            activations.append(x.cpu().numpy().flatten())
            layer_names.append(f'Layer {i+1}')
    
    n_layers = len(activations)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    
    if n_layers == 1:
        axes = [axes]
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_layers))
    
    for ax, act, name, color in zip(axes, activations, layer_names, colors):
        ax.hist(act, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'{name}\nμ={np.mean(act):.3f}, σ={np.std(act):.3f}')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Activation Distributions ({model.activation_name})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_neuron_responses(model, X, layer_idx=0, figsize=(12, 8)):
    """
    Show what each neuron in a layer responds to (2D input space).
    
    Args:
        model: PyTorch neural network model
        X: Input tensor for determining bounds
        layer_idx: Which hidden layer to visualize
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
    y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    
    model.eval()
    with torch.no_grad():
        x = grid_tensor
        for i, (layer, bn) in enumerate(zip(model.layers[:layer_idx+1], model.batch_norms[:layer_idx+1])):
            x = layer(x)
            x = bn(x)
            x = model.activation(x)
        
        activations = x.cpu().numpy()
    
    n_neurons = activations.shape[1]
    cols = min(8, n_neurons)
    rows = (n_neurons + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n_neurons:
                z = activations[:, idx].reshape(xx.shape)
                im = axes[i, j].contourf(xx, yy, z, levels=20, cmap='viridis')
                axes[i, j].set_title(f'Neuron {idx+1}', fontsize=8)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
            else:
                axes[i, j].axis('off')
    
    plt.suptitle(f'Neuron Response Maps - Layer {layer_idx + 1}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_3d_decision_surface(model, X, y, figsize=(10, 8), elevation=30, azimuth=45):
    """
    Create 3D visualization of the decision surface.
    
    Args:
        model: PyTorch neural network model
        X: Input data (numpy array)
        y: Labels
        figsize: Figure size tuple
        elevation: Viewing angle elevation
        azimuth: Viewing angle azimuth
    
    Returns:
        matplotlib figure
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    
    model.eval()
    with torch.no_grad():
        predictions = model(grid_tensor)
        probs = torch.softmax(predictions, dim=1)[:, 1].numpy()
    
    zz = probs.reshape(xx.shape)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the decision surface
    surf = ax.plot_surface(xx, yy, zz, cmap='RdYlBu', alpha=0.7, 
                           linewidth=0, antialiased=True)
    
    # Plot decision boundary plane at z=0.5
    ax.contour(xx, yy, zz, levels=[0.5], colors='black', linewidths=2)
    
    # Scatter plot of data points
    colors = ['red' if label == 0 else 'blue' for label in y]
    ax.scatter(X[:, 0], X[:, 1], y * 0.02 + 0.5, c=colors, s=30, 
               edgecolors='white', linewidth=0.5, alpha=0.8)
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_zlabel('P(Class 1)')
    ax.set_title('3D Decision Surface', fontsize=12, fontweight='bold')
    ax.view_init(elev=elevation, azim=azimuth)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Probability')
    plt.tight_layout()
    return fig


def plot_training_comparison(histories: List[Dict], labels: List[str], figsize=(14, 5)):
    """
    Compare training histories from multiple runs.
    
    Args:
        histories: List of history dictionaries (each with 'train_loss', 'val_loss', etc.)
        labels: List of labels for each run
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    # Loss comparison
    for hist, label, color in zip(histories, labels, colors):
        if 'train_loss' in hist:
            axes[0].plot(hist['train_loss'], label=f'{label} (train)', 
                        color=color, linestyle='-', alpha=0.8)
        if 'val_loss' in hist:
            axes[0].plot(hist['val_loss'], label=f'{label} (val)', 
                        color=color, linestyle='--', alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    for hist, label, color in zip(histories, labels, colors):
        if 'train_acc' in hist:
            axes[1].plot([a*100 for a in hist['train_acc']], label=f'{label} (train)', 
                        color=color, linestyle='-', alpha=0.8)
        if 'val_acc' in hist:
            axes[1].plot([a*100 for a in hist['val_acc']], label=f'{label} (val)', 
                        color=color, linestyle='--', alpha=0.8)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Comparison')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    # Final metrics bar chart
    metrics_data = []
    for hist in histories:
        if 'val_acc' in hist and hist['val_acc']:
            metrics_data.append({
                'final_acc': hist['val_acc'][-1] * 100,
                'best_acc': max(hist['val_acc']) * 100,
                'final_loss': hist['val_loss'][-1] if 'val_loss' in hist else 0
            })
        else:
            metrics_data.append({'final_acc': 0, 'best_acc': 0, 'final_loss': 0})
    
    x = np.arange(len(labels))
    width = 0.35
    
    axes[2].bar(x - width/2, [m['final_acc'] for m in metrics_data], width, 
               label='Final Acc', color='steelblue', alpha=0.8)
    axes[2].bar(x + width/2, [m['best_acc'] for m in metrics_data], width, 
               label='Best Acc', color='seagreen', alpha=0.8)
    
    axes[2].set_xlabel('Experiment')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Final vs Best Accuracy')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Training History Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_learning_curve(history: Dict, figsize=(12, 4)):
    """
    Plot comprehensive learning curves.
    
    Args:
        history: Training history dictionary
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss curves
    if 'train_loss' in history:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r--', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Fill between for overfitting detection
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].fill_between(epochs, history['train_loss'], history['val_loss'], 
                            alpha=0.2, color='purple')
    
    # Accuracy curves
    if 'train_acc' in history:
        axes[1].plot(epochs, [a*100 for a in history['train_acc']], 'g-', 
                    label='Train', linewidth=2)
    if 'val_acc' in history:
        axes[1].plot(epochs, [a*100 for a in history['val_acc']], 'orange', 
                    linestyle='--', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Precision/Recall/F1
    if 'precision' in history and history['precision']:
        axes[2].plot(epochs, [p*100 for p in history['precision']], 
                    label='Precision', linewidth=2)
    if 'recall' in history and history['recall']:
        axes[2].plot(epochs, [r*100 for r in history['recall']], 
                    label='Recall', linewidth=2)
    if 'f1' in history and history['f1']:
        axes[2].plot(epochs, [f*100 for f in history['f1']], 
                    label='F1 Score', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score (%)')
    axes[2].set_title('Precision/Recall/F1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Learning Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_weight_distribution_history(weight_history: List[Dict], figsize=(12, 6)):
    """
    Visualize how weight distributions change during training.
    
    Args:
        weight_history: List of weight snapshots (each is output of model.get_layer_weights())
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    if not weight_history:
        return None
    
    n_layers = len(weight_history[0])
    n_snapshots = len(weight_history)
    
    fig, axes = plt.subplots(n_layers, 1, figsize=figsize)
    
    if n_layers == 1:
        axes = [axes]
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_snapshots))
    
    for layer_idx, ax in enumerate(axes):
        for snap_idx, (weights, color) in enumerate(zip(weight_history, colors)):
            w = weights[layer_idx]['weight'].flatten()
            ax.hist(w, bins=50, alpha=0.3, color=color, 
                   label=f'Epoch {snap_idx * (100 // n_snapshots)}' if snap_idx % 2 == 0 else None)
        
        ax.set_title(f'Layer {layer_idx + 1} Weight Distribution Over Time')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weight Distribution Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_network_diagram(layer_sizes: List[int], figsize=(10, 6)):
    """
    Create a visual diagram of the network architecture.
    
    Args:
        layer_sizes: List of neurons per layer [input, hidden1, hidden2, ..., output]
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    
    # Calculate positions
    layer_x = np.linspace(0.1, 0.9, n_layers)
    
    neuron_positions = []
    for i, size in enumerate(layer_sizes):
        y_positions = np.linspace(0.1, 0.9, min(size, 10))
        if size > 10:
            y_positions = np.linspace(0.1, 0.9, 10)
        neuron_positions.append([(layer_x[i], y) for y in y_positions])
    
    # Draw connections
    for i in range(n_layers - 1):
        for pos1 in neuron_positions[i]:
            for pos2 in neuron_positions[i + 1]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                       'gray', alpha=0.2, linewidth=0.5)
    
    # Draw neurons
    colors = ['#3498db', '#2ecc71', '#2ecc71', '#2ecc71', '#e74c3c']
    
    for i, (positions, size) in enumerate(zip(neuron_positions, layer_sizes)):
        color = colors[min(i, len(colors)-1)]
        if i == n_layers - 1:
            color = '#e74c3c'
        
        for pos in positions:
            circle = plt.Circle(pos, 0.02, color=color, ec='black', linewidth=1.5)
            ax.add_patch(circle)
        
        # Add "..." if too many neurons
        if size > 10:
            ax.text(layer_x[i], 0.5, '...', fontsize=16, ha='center', va='center')
        
        # Layer label
        if i == 0:
            label = f'Input\n({size})'
        elif i == n_layers - 1:
            label = f'Output\n({size})'
        else:
            label = f'Hidden {i}\n({size})'
        
        ax.text(layer_x[i], -0.05, label, fontsize=10, ha='center', va='top')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Network Architecture', fontsize=14, fontweight='bold')
    
    return fig

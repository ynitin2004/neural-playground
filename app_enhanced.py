"""
Neural Network Playground - Enhanced Version with Advanced Visualizations
Features:
- Multiple activation functions (ReLU, Tanh, Sigmoid, LeakyReLU, GELU, etc.)
- Multiple optimizers (Adam, SGD, RMSprop, AdaGrad, AdamW)
- Regularization (L1/L2, Dropout)
- Batch Normalization
- Live training metrics (Accuracy, Precision, Recall, F1)
- Weight matrix heatmaps
- Gradient flow visualization  
- Neuron activation maps
- 3D decision surface
- Training history comparison
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

from dataset import (
    classify_two_gauss_data,
    regress_gaussian,
    classify_spiral_data,
    classify_circle_data,
    classify_xor_data
)
from models import (
    EnhancedANN,
    get_optimizer,
    MetricsTracker,
    compute_metrics
)
from visualizations import (
    plot_weight_heatmaps,
    plot_gradient_flow,
    plot_activation_distribution,
    plot_neuron_responses,
    plot_3d_decision_surface,
    plot_training_comparison,
    plot_learning_curve,
    create_network_diagram
)

# Page configuration
st.set_page_config(
    page_title="Neural Network Playground",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00ff88;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
    }
    .stProgress > div > div > div > div {
        background-color: #00ff88;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #262730;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def load_dataset(choice, num_samples, noise):
    """Load the selected dataset."""
    datasets = {
        'Two Gaussian Clusters': classify_two_gauss_data,
        'Gaussian Regression': regress_gaussian,
        'Spiral Data': classify_spiral_data,
        'Circle Data': classify_circle_data,
        'XOR Data': classify_xor_data
    }
    return datasets[choice](num_samples, noise)


def plot_decision_boundary(model, X, points, ax):
    """Plot decision boundary with data points."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    
    model.eval()
    with torch.no_grad():
        predictions = model(grid_tensor).numpy()
    model.train()
    
    zz = np.argmax(predictions, axis=1).reshape(xx.shape)
    
    ax.contourf(xx, yy, zz, alpha=0.6, cmap='RdYlBu')
    
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    labels = [p.label for p in points]
    ax.scatter(x_coords, y_coords, c=labels, cmap='coolwarm', s=30, edgecolors='white', linewidth=0.5)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def train_model_with_viz(model, X_train, y_train, X_val, y_val, config, 
                         progress_bar, metrics_placeholder, plot_placeholder, 
                         points, X, collect_weights=False):
    """Train the model with live updates and collect visualization data."""
    
    optimizer = get_optimizer(
        config['optimizer'], 
        model.parameters(), 
        config['learning_rate'],
        config['l2_lambda'] if config['reg_type'] == 'L2' else 0.0
    )
    
    loss_function = nn.CrossEntropyLoss()
    tracker = MetricsTracker()
    
    num_classes = len(torch.unique(y_train))
    weight_history = []
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for epoch in range(config['epochs']):
        model.train()
        
        # Forward pass
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        
        # Add regularization
        if config['reg_type'] == 'L1':
            loss += model.l1_regularization(config['l1_lambda'])
        elif config['reg_type'] == 'L2' and config['l2_lambda'] > 0:
            loss += model.l2_regularization(config['l2_lambda'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Collect weight snapshots periodically
        if collect_weights and (epoch % (config['epochs'] // 10) == 0 or epoch == config['epochs'] - 1):
            weight_history.append(model.get_layer_weights())
        
        # Compute metrics
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train).argmax(dim=1)
            val_pred = model(X_val).argmax(dim=1)
            val_loss = loss_function(model(X_val), y_val).item()
        
        train_metrics = compute_metrics(y_train, train_pred, num_classes)
        val_metrics = compute_metrics(y_val, val_pred, num_classes)
        
        tracker.update(
            train_loss=loss.item(),
            val_loss=val_loss,
            train_acc=train_metrics['accuracy'],
            val_acc=val_metrics['accuracy'],
            precision=val_metrics['precision'],
            recall=val_metrics['recall'],
            f1=val_metrics['f1'],
            lr=config['learning_rate']
        )
        
        progress_bar.progress((epoch + 1) / config['epochs'])
        
        if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
            latest = tracker.get_latest()
            
            with metrics_placeholder.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Train Loss", f"{latest['train_loss']:.4f}")
                with col2:
                    st.metric("Val Loss", f"{latest['val_loss']:.4f}")
                with col3:
                    st.metric("Accuracy", f"{latest['val_acc']*100:.1f}%")
                with col4:
                    st.metric("Precision", f"{latest['precision']*100:.1f}%")
                with col5:
                    st.metric("F1 Score", f"{latest['f1']*100:.1f}%")
            
            for ax in axes:
                ax.clear()
            
            history = tracker.get_history()
            
            plot_decision_boundary(model, X, points, axes[0])
            axes[0].set_title(f'Decision Boundary (Epoch {epoch + 1})')
            
            axes[1].plot(history['train_loss'], label='Train Loss', color='blue')
            axes[1].plot(history['val_loss'], label='Val Loss', color='red')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Loss Curves')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot([a*100 for a in history['train_acc']], label='Train Acc', color='green')
            axes[2].plot([a*100 for a in history['val_acc']], label='Val Acc', color='orange')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Accuracy (%)')
            axes[2].set_title('Accuracy Curves')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_placeholder.pyplot(fig)
    
    plt.close(fig)
    return model, tracker, weight_history


def main():
    st.title('🧠 Neural Network Playground')
    st.markdown("*Interactive visualization of neural network training with advanced features*")
    
    # Initialize session state
    if 'experiment_history' not in st.session_state:
        st.session_state.experiment_history = []
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    
    # Sidebar configuration
    st.sidebar.header('📊 Dataset Configuration')
    
    dataset_name = st.sidebar.selectbox(
        'Select Dataset',
        ['Two Gaussian Clusters', 'Spiral Data', 'Circle Data', 'XOR Data', 'Gaussian Regression']
    )
    
    num_samples = st.sidebar.slider('Number of Samples', 100, 2000, 500, step=100)
    noise = st.sidebar.slider('Noise Level', 0.0, 1.0, 0.1, step=0.05)
    
    st.sidebar.header('🏗️ Network Architecture')
    
    num_hidden_layers = st.sidebar.slider('Number of Hidden Layers', 1, 5, 2)
    
    hidden_layers = []
    for i in range(num_hidden_layers):
        neurons = st.sidebar.slider(f'Neurons in Layer {i+1}', 4, 64, 20, step=4)
        hidden_layers.append(neurons)
    
    activation = st.sidebar.selectbox(
        'Activation Function',
        ['ReLU', 'Tanh', 'Sigmoid', 'LeakyReLU', 'GELU', 'ELU', 'SELU', 'Swish']
    )
    
    st.sidebar.header('⚙️ Training Configuration')
    
    epochs = st.sidebar.slider('Number of Epochs', 50, 500, 200, step=50)
    learning_rate = st.sidebar.select_slider(
        'Learning Rate',
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        value=0.01
    )
    
    optimizer_name = st.sidebar.selectbox(
        'Optimizer',
        ['Adam', 'AdamW', 'SGD', 'RMSprop', 'AdaGrad']
    )
    
    st.sidebar.header('🛡️ Regularization')
    
    use_batch_norm = st.sidebar.checkbox('Batch Normalization', value=False)
    dropout_rate = st.sidebar.slider('Dropout Rate', 0.0, 0.5, 0.0, step=0.1)
    
    reg_type = st.sidebar.selectbox('Regularization Type', ['None', 'L1', 'L2'])
    
    l1_lambda = 0.0
    l2_lambda = 0.0
    if reg_type == 'L1':
        l1_lambda = st.sidebar.select_slider('L1 Lambda', options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
    elif reg_type == 'L2':
        l2_lambda = st.sidebar.select_slider('L2 Lambda', options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
    
    # Experiment name for comparison
    st.sidebar.header('📝 Experiment')
    experiment_name = st.sidebar.text_input('Experiment Name', f'Exp_{len(st.session_state.experiment_history)+1}')
    
    # Architecture summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Model Summary")
    arch_str = f"Input(2) → "
    for i, h in enumerate(hidden_layers):
        arch_str += f"Dense({h}) → "
    arch_str += "Output(2)"
    st.sidebar.text(arch_str)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎮 Training", 
        "🔍 Model Analysis", 
        "📊 Comparisons",
        "🎨 3D Visualization"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("📈 Live Metrics")
            metrics_placeholder = st.empty()
            
            with metrics_placeholder.container():
                mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
                with mcol1:
                    st.metric("Train Loss", "—")
                with mcol2:
                    st.metric("Val Loss", "—")
                with mcol3:
                    st.metric("Accuracy", "—")
                with mcol4:
                    st.metric("Precision", "—")
                with mcol5:
                    st.metric("F1 Score", "—")
            
            # Network architecture diagram
            st.subheader("🏗️ Network Architecture")
            layer_sizes = [2] + hidden_layers + [2]
            arch_fig = create_network_diagram(layer_sizes, figsize=(8, 4))
            st.pyplot(arch_fig)
            plt.close(arch_fig)
        
        with col1:
            plot_placeholder = st.empty()
            
            if 'points' not in st.session_state:
                points = load_dataset(dataset_name, num_samples, noise)
                st.session_state.points = points
            else:
                points = st.session_state.points
            
            fig_init, ax_init = plt.subplots(figsize=(8, 6))
            x_coords = [p.x for p in points]
            y_coords = [p.y for p in points]
            labels = [p.label for p in points]
            ax_init.scatter(x_coords, y_coords, c=labels, cmap='coolwarm', s=30, edgecolors='white', linewidth=0.5)
            ax_init.set_title('Dataset Preview')
            ax_init.grid(True, alpha=0.3)
            plot_placeholder.pyplot(fig_init)
            plt.close(fig_init)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            load_btn = st.button('🔄 Load New Data', use_container_width=True)
        
        with col_btn2:
            train_btn = st.button('🚀 Train Model', type='primary', use_container_width=True)
        
        if load_btn:
            st.session_state.points = load_dataset(dataset_name, num_samples, noise)
            st.rerun()
        
        if train_btn:
            points = load_dataset(dataset_name, num_samples, noise)
            st.session_state.points = points
            
            X = np.array([[p.x, p.y] for p in points])
            y = np.array([p.label for p in points])
            
            unique_labels = np.unique(y)
            num_classes = len(unique_labels)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_t = torch.FloatTensor(X_train)
            X_val_t = torch.FloatTensor(X_val)
            y_train_t = torch.LongTensor(y_train)
            y_val_t = torch.LongTensor(y_val)
            
            model = EnhancedANN(
                input_features=2,
                hidden_layers=hidden_layers,
                out_features=num_classes,
                activation=activation,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            )
            
            config = {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'optimizer': optimizer_name,
                'reg_type': reg_type,
                'l1_lambda': l1_lambda,
                'l2_lambda': l2_lambda
            }
            
            progress_bar = st.progress(0)
            st.markdown("**Training in progress...**")
            
            start_time = time.time()
            model, tracker, weight_history = train_model_with_viz(
                model, X_train_t, y_train_t, X_val_t, y_val_t, 
                config, progress_bar, metrics_placeholder, plot_placeholder, 
                points, X, collect_weights=True
            )
            elapsed = time.time() - start_time
            
            # Store for visualizations
            st.session_state.trained_model = model
            st.session_state.training_data = {
                'X_train': X_train_t,
                'X_val': X_val_t,
                'y_train': y_train_t,
                'y_val': y_val_t,
                'X': X,
                'y': y,
                'points': points,
                'weight_history': weight_history
            }
            
            # Save to experiment history
            history = tracker.get_history()
            st.session_state.experiment_history.append({
                'name': experiment_name,
                'history': history,
                'config': config,
                'architecture': hidden_layers.copy(),
                'activation': activation
            })
            
            st.success(f"✅ Training completed in {elapsed:.1f} seconds!")
            
            # Final results
            st.subheader("📊 Final Results")
            final_col1, final_col2, final_col3, final_col4 = st.columns(4)
            
            with final_col1:
                st.metric("Final Train Accuracy", f"{history['train_acc'][-1]*100:.2f}%")
            with final_col2:
                st.metric("Final Val Accuracy", f"{history['val_acc'][-1]*100:.2f}%")
            with final_col3:
                st.metric("Best Val Accuracy", f"{max(history['val_acc'])*100:.2f}%")
            with final_col4:
                st.metric("Final F1 Score", f"{history['f1'][-1]*100:.2f}%")
    
    with tab2:
        st.header("🔍 Model Analysis")
        
        if st.session_state.trained_model is None:
            st.info("👆 Train a model first to see the analysis!")
        else:
            model = st.session_state.trained_model
            data = st.session_state.training_data
            
            analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
                "⚖️ Weights", "📉 Gradients", "🧠 Activations", "📈 Learning Curves"
            ])
            
            with analysis_tab1:
                st.subheader("Weight Matrix Heatmaps")
                st.markdown("*Visualize the learned weight matrices in each layer*")
                
                weight_fig = plot_weight_heatmaps(model, figsize=(14, 4))
                if weight_fig:
                    st.pyplot(weight_fig)
                    plt.close(weight_fig)
            
            with analysis_tab2:
                st.subheader("Gradient Flow Analysis")
                st.markdown("*Check for vanishing or exploding gradients*")
                
                # Do a forward-backward pass to get gradients
                model.train()
                y_pred = model(data['X_train'])
                loss = nn.CrossEntropyLoss()(y_pred, data['y_train'])
                loss.backward()
                
                grad_fig = plot_gradient_flow(model, figsize=(12, 5))
                if grad_fig:
                    st.pyplot(grad_fig)
                    plt.close(grad_fig)
                
                model.zero_grad()
            
            with analysis_tab3:
                st.subheader("Neuron Activation Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Activation Distribution per Layer**")
                    act_fig = plot_activation_distribution(model, data['X_train'], figsize=(14, 4))
                    if act_fig:
                        st.pyplot(act_fig)
                        plt.close(act_fig)
                
                with col2:
                    st.markdown("**Neuron Response Maps**")
                    layer_to_viz = st.selectbox("Select Layer", range(len(model.layers)), format_func=lambda x: f"Layer {x+1}")
                    
                    neuron_fig = plot_neuron_responses(model, data['X_train'], layer_idx=layer_to_viz, figsize=(12, 8))
                    if neuron_fig:
                        st.pyplot(neuron_fig)
                        plt.close(neuron_fig)
            
            with analysis_tab4:
                st.subheader("Learning Curves")
                
                if st.session_state.experiment_history:
                    latest_exp = st.session_state.experiment_history[-1]
                    learning_fig = plot_learning_curve(latest_exp['history'], figsize=(14, 4))
                    if learning_fig:
                        st.pyplot(learning_fig)
                        plt.close(learning_fig)
    
    with tab3:
        st.header("📊 Experiment Comparison")
        
        if len(st.session_state.experiment_history) < 1:
            st.info("👆 Train at least one model to see comparisons!")
        else:
            st.markdown(f"**{len(st.session_state.experiment_history)} experiments recorded**")
            
            # Show experiment table
            exp_data = []
            for exp in st.session_state.experiment_history:
                exp_data.append({
                    'Name': exp['name'],
                    'Architecture': str(exp['architecture']),
                    'Activation': exp['activation'],
                    'Optimizer': exp['config']['optimizer'],
                    'LR': exp['config']['learning_rate'],
                    'Final Acc': f"{exp['history']['val_acc'][-1]*100:.1f}%",
                    'Best Acc': f"{max(exp['history']['val_acc'])*100:.1f}%"
                })
            
            st.dataframe(exp_data, use_container_width=True)
            
            if len(st.session_state.experiment_history) >= 2:
                st.subheader("Training Comparison")
                
                histories = [exp['history'] for exp in st.session_state.experiment_history]
                labels = [exp['name'] for exp in st.session_state.experiment_history]
                
                comp_fig = plot_training_comparison(histories, labels, figsize=(14, 5))
                if comp_fig:
                    st.pyplot(comp_fig)
                    plt.close(comp_fig)
            
            if st.button("🗑️ Clear Experiment History"):
                st.session_state.experiment_history = []
                st.rerun()
    
    with tab4:
        st.header("🎨 3D Decision Surface")
        
        if st.session_state.trained_model is None:
            st.info("👆 Train a model first to see the 3D visualization!")
        else:
            model = st.session_state.trained_model
            data = st.session_state.training_data
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("**View Controls**")
                elevation = st.slider("Elevation", 0, 90, 30)
                azimuth = st.slider("Azimuth", 0, 360, 45)
            
            with col2:
                surface_fig = plot_3d_decision_surface(
                    model, data['X'], data['y'], 
                    figsize=(10, 8),
                    elevation=elevation,
                    azimuth=azimuth
                )
                if surface_fig:
                    st.pyplot(surface_fig)
                    plt.close(surface_fig)
            
            st.markdown("""
            **📖 How to interpret:**
            - The surface shows the probability of Class 1
            - Blue regions → High probability of Class 1
            - Red regions → Low probability of Class 1 (High probability of Class 0)
            - The black line shows the decision boundary (50% probability)
            """)


if __name__ == '__main__':
    main()

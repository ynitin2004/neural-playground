import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from dataset import (
    classify_two_gauss_data,
    regress_gaussian,
    classify_spiral_data,
    classify_circle_data,
    classify_xor_data
)
import os

class ANN_Model(nn.Module):
    def __init__(self, input_features, hidden_layers, out_features):
        super(ANN_Model, self).__init__()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_features, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        self.out = nn.Linear(hidden_layers[-1], out_features)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x

def load_dataset(choice, num_samples, noise):
    if choice == 'Two Gaussian Clusters':
        return classify_two_gauss_data(num_samples, noise)
    elif choice == 'Gaussian Regression':
        return regress_gaussian(num_samples, noise)
    elif choice == 'Spiral Data':
        return classify_spiral_data(num_samples, noise)
    elif choice == 'Circle Data':
        return classify_circle_data(num_samples, noise)
    elif choice == 'XOR Data':
        return classify_xor_data(num_samples, noise)
    else:
        raise ValueError("Invalid choice. Please choose a valid dataset.")

def plot_points(ax, points):
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    labels = [p.label for p in points]
    scatter = ax.scatter(x_coords, y_coords, c=labels, cmap='coolwarm', s=20)
    return scatter

def plot_heatmap(model, ax, x_min, x_max, y_min, y_max):
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    with torch.no_grad():
        predictions = model(grid_tensor).numpy()
    zz = np.argmax(predictions, axis=1).reshape(xx.shape)
    heatmap = ax.imshow(zz, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis', alpha=0.5, aspect='auto')
    return heatmap

def animate(i, model, X, points, ax, scatter):
    ax.clear()
    plot_points(ax, points)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    plot_heatmap(model, ax, x_min, x_max, y_min, y_max)
    ax.set_title(f'Epoch {i * 10}')
    return scatter,

# Streamlit UI
st.title('Neural Network Playground')

dataset_name = st.sidebar.selectbox('Select Dataset', [
    'Two Gaussian Clusters', 'Gaussian Regression', 'Spiral Data', 
    'Circle Data', 'XOR Data'
])
num_samples = st.sidebar.slider('Number of Samples', 100, 1000, step=100)
noise = st.sidebar.slider('Noise Level', 0.0, 1.0, step=0.1)
hidden_layers_input = st.sidebar.text_input('Hidden Layers (comma separated)', '20,20')
epochs = st.sidebar.slider('Number of Epochs', 50, 500, step=50)

# Initialize the final_losses variable outside the button click
final_losses = []

if st.sidebar.button('Load Data'):
    points = load_dataset(dataset_name, num_samples, noise)

    # Data Preparation
    X = np.array([[p.x, p.y] for p in points])
    y = np.array([p.label for p in points])

    # Ensure labels are zero-based integers
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Neural Network Configuration
    input_features = X_train.shape[1]
    hidden_layers = [int(x) for x in hidden_layers_input.split(',')]
    out_features = num_classes

    model = ANN_Model(input_features, hidden_layers, out_features)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    fig, ax = plt.subplots(figsize=(10, 5))  # Increase size to make it clearer
    scatter = plot_points(ax, points)
    
    def update_plot(epoch):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        final_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            animate(epoch // 10, model, X, points, ax, scatter)
    
    anim = FuncAnimation(fig, update_plot, frames=range(epochs), repeat=False, interval=500)  # Adjust interval for faster updates
    anim.save('animation.mp4', writer='ffmpeg')  # Save as MP4 for faster rendering
    
    st.video('animation.mp4')  # Display the video

    # Show final loss plot
    plt.figure()
    plt.plot(final_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    st.pyplot(plt)




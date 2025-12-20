# 🧠 Neural Network Playground

An interactive visualization tool for understanding neural network training, built with PyTorch and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green.svg)

## ✨ Features

### 📊 Datasets
- Two Gaussian Clusters
- Spiral Data
- Circle Data
- XOR Data
- Gaussian Regression

### 🏗️ Network Architecture
- Configurable hidden layers (1-5 layers)
- Adjustable neurons per layer (4-64)
- Multiple activation functions:
  - ReLU, Tanh, Sigmoid
  - LeakyReLU, GELU, ELU
  - SELU, Swish

### ⚙️ Training Options
- **Optimizers**: Adam, AdamW, SGD, RMSprop, AdaGrad
- **Learning Rate**: Configurable (0.0001 - 0.1)
- **Epochs**: 50 - 500

### 🛡️ Regularization
- **Dropout**: 0% - 50%
- **L1 Regularization** (Lasso)
- **L2 Regularization** (Ridge/Weight Decay)
- **Batch Normalization**

### 📈 Live Metrics
- Training & Validation Loss
- Training & Validation Accuracy
- Precision, Recall, F1 Score
- Real-time decision boundary visualization

### 🔍 Advanced Visualizations (NEW!)
- **Weight Heatmaps**: Visualize learned weight matrices per layer
- **Gradient Flow**: Detect vanishing/exploding gradients
- **Neuron Response Maps**: See what each neuron responds to
- **Activation Distributions**: Analyze activation patterns
- **3D Decision Surface**: Interactive 3D probability surface
- **Experiment Comparison**: Compare multiple training runs
- **Network Architecture Diagram**: Visual representation of your model

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ynitin2004/neural-playground.git
cd neural-playground

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
# Run the enhanced version
streamlit run app_enhanced.py

# Or run the original version
streamlit run app.py
```

## 📁 Project Structure

```
neural-playground/
├── app.py              # Original Streamlit app
├── app_enhanced.py     # Enhanced version with all features
├── models.py           # Neural network models
├── dataset.py          # Dataset generators
├── training_utils.py   # Training utilities
├── visualizations.py   # Advanced visualization functions
├── nn1.py              # Command-line version
├── heatmap1.py         # Heatmap visualization
└── README.md
```

## 🎮 Usage Guide

1. **Select Dataset**: Choose from 5 different 2D classification datasets
2. **Configure Network**: Adjust layers, neurons, and activation functions
3. **Set Training Options**: Choose optimizer, learning rate, and epochs
4. **Add Regularization**: Enable dropout, batch norm, or L1/L2 regularization
5. **Train**: Click "Train Model" and watch the live visualization
6. **Analyze**: Explore the Model Analysis tab for deep insights
7. **Compare**: Run multiple experiments and compare results
8. **Visualize 3D**: View interactive 3D decision surfaces

## 📸 Features

### 🎮 Training Tab
- Live decision boundary updates
- Real-time loss and accuracy curves
- Network architecture diagram

### 🔍 Model Analysis Tab
- **Weights**: Heatmaps of weight matrices
- **Gradients**: Gradient flow analysis for debugging
- **Activations**: Distribution and response maps
- **Learning Curves**: Comprehensive training metrics

### 📊 Comparisons Tab
- Side-by-side experiment comparison
- Training history overlay
- Best accuracy tracking

### 🎨 3D Visualization Tab
- Interactive 3D decision surface
- Adjustable view angles
- Probability surface visualization

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.0+
- NumPy
- Matplotlib
- scikit-learn

## 📄 License

MIT License - feel free to use this project for learning and experimentation!

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

Made with ❤️ for learning neural networks

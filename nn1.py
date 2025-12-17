import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataset import (classify_two_gauss_data, regress_gaussian,
                     classify_spiral_data, classify_circle_data, classify_xor_data)
from sklearn.metrics import accuracy_score
from matplotlib.animation import FuncAnimation

def get_dataset_choice():
    print("Choose a dataset:")
    print("1: Two Gaussian Clusters")
    print("2: Gaussian Regression")
    print("3: Spiral Data")
    print("4: Circle Data")
    print("5: XOR Data")
    choice = int(input("Enter the number of the dataset you want to use: "))
    return choice

def load_dataset(choice, num_samples, noise):
    if choice == 1:
        return classify_two_gauss_data(num_samples, noise)
    elif choice == 2:
        return regress_gaussian(num_samples, noise)
    elif choice == 3:
        return classify_spiral_data(num_samples, noise)
    elif choice == 4:
        return classify_circle_data(num_samples, noise)
    elif choice == 5:
        return classify_xor_data(num_samples, noise)
    else:
        raise ValueError("Invalid choice. Please choose a number between 1 and 5.")

def prepare_data(choice, num_samples, noise):
    points = load_dataset(choice, num_samples, noise)
    X = np.array([[p.x, p.y] for p in points])
    y = np.array([p.label for p in points])
    y = np.where(y == -1, 0, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    return X_train, X_test, y_train, y_test

class ANN_Model(nn.Module):
    def __init__(self, input_features=2, hidden_layers=[20, 20], out_features=2):
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

def create_model(input_features, num_hidden_layers, hidden_layer_sizes, out_features):
    hidden_layers = hidden_layer_sizes[:num_hidden_layers]
    return ANN_Model(input_features=input_features, hidden_layers=hidden_layers, out_features=out_features)

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = torch.FloatTensor(grid)
    with torch.no_grad():
        predictions = model(grid).argmax(dim=1).numpy()
    predictions = predictions.reshape(xx.shape)
    return xx, yy, predictions

def train_and_plot_model(model, X_train, y_train, X_test, y_test, epochs, loss_function, optimizer):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, edgecolors='k', cmap='coolwarm')
    ax2.set_xlim(0, epochs)
    ax2.set_ylim(0, 2)
    line, = ax2.plot([], [], lw=2)
    final_losses = []

    def update_plot(epoch):
        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        if epoch < len(final_losses):
            final_losses[epoch] = loss.item()
        else:
            final_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            xx, yy, predictions = plot_decision_boundary(model, X_train.numpy(), y_train.numpy())
            ax1.clear()
            ax1.contourf(xx, yy, predictions, alpha=0.3, cmap='coolwarm')
            scatter = ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, edgecolors='k', cmap='coolwarm')
            predictions = model(X_test).argmax(dim=1).numpy()
            score = accuracy_score(y_test, predictions)
            print(f"Epoch number: {epoch + 1} and the loss: {loss.item()}")
            print(f"Accuracy after epoch {epoch + 1}: {score}")

        line.set_data(range(len(final_losses)), final_losses)
        ax2.relim()
        ax2.autoscale_view()

    ani = FuncAnimation(fig, update_plot, frames=range(epochs), repeat=False)
    plt.show()

def main():
    choice = get_dataset_choice()
    num_samples = 100
    noise = 0.5
    X_train, X_test, y_train, y_test = prepare_data(choice, num_samples, noise)

    input_features = 2
    num_hidden_layers = int(input("Enter the number of hidden layers: "))
    hidden_layer_sizes = [int(input(f"Enter the size of hidden layer {i+1}: ")) for i in range(num_hidden_layers)]
    out_features = 2

    model = create_model(input_features, num_hidden_layers, hidden_layer_sizes, out_features)
    print(model)

    torch.manual_seed(20)
    model = create_model(input_features, num_hidden_layers, hidden_layer_sizes, out_features)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = int(input("Enter the number of epochs: "))

    train_and_plot_model(model, X_train, y_train, X_test, y_test, epochs, loss_function, optimizer)

if __name__ == "__main__":
    main()

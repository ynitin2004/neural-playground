import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons

class Example2D:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

def shuffle(array):
    np.random.shuffle(array)

def classify_two_gauss_data(num_samples, noise):
    centers = [[2, 2], [-2, -2]]
    X, y = make_blobs(n_samples=num_samples, centers=centers, cluster_std=noise)
    points = [Example2D(x, y, label) for (x, y), label in zip(X, y)]
    return points

def regress_plane(num_samples, noise):
    radius = 6
    X = np.random.uniform(-radius, radius, size=(num_samples, 2))
    y = (X[:, 0] + X[:, 1]) / 10
    y += noise * np.random.uniform(-radius, radius, size=num_samples)
    points = [Example2D(x, y, label) for (x, y), label in zip(X, y)]
    return points

def regress_gaussian(num_samples, noise):
    gaussians = [
        [-4, 2.5, 1],
        [0, 2.5, -1],
        [4, 2.5, 1],
        [-4, -2.5, -1],
        [0, -2.5, 1],
        [4, -2.5, -1]
    ]
    points = []
    radius = 6
    for _ in range(num_samples):
        x, y = np.random.uniform(-radius, radius, size=2)
        label = max([sign * np.exp(-((x - cx) ** 2 + (y - cy) ** 2)) for cx, cy, sign in gaussians])
        x += noise * np.random.uniform(-radius, radius)
        y += noise * np.random.uniform(-radius, radius)
        points.append(Example2D(x, y, label))
    return points

def classify_spiral_data(num_samples, noise):
    points = []
    n = num_samples // 2
    for delta_t, label in [(0, 1), (np.pi, -1)]:
        for i in range(n):
            r = i / n * 5
            t = 1.75 * i / n * 2 * np.pi + delta_t
            x = r * np.sin(t) 
            y = r * np.cos(t) 
            points.append(Example2D(x, y, label))
    return points

def classify_circle_data(num_samples, noise):
    points = []
    radius = 5
    for _ in range(num_samples // 2):
        r = np.random.uniform(0, radius * 0.5)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        label = 1 if (x ** 2 + y ** 2) < (radius * 0.5) ** 2 else -1
        x += noise * np.random.uniform(-radius, radius)
        y += noise * np.random.uniform(-radius, radius)
        points.append(Example2D(x, y, label))
    for _ in range(num_samples // 2):
        r = np.random.uniform(radius * 0.7, radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        label = 1 if (x ** 2 + y ** 2) < (radius * 0.5) ** 2 else -1
        x += noise * np.random.uniform(-radius, radius)
        y += noise * np.random.uniform(-radius, radius)
        points.append(Example2D(x, y, label))
    return points

def classify_xor_data(num_samples, noise):
    points = []
    for _ in range(num_samples):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        label = 1 if x * y >= 0 else -1
        x += noise * np.random.uniform(-5, 5)
        y += noise * np.random.uniform(-5, 5)
        points.append(Example2D(x, y, label))
    return points

def plot_points(points):
    x_coords = [p.x for p in points]
    y_coords = [p.y for p in points]
    labels = [p.label for p in points]
    plt.scatter(x_coords, y_coords, c=labels, cmap='coolwarm')
    plt.show()

# Example usage
#points = classify_two_gauss_data(100, 0.5)
#plot_points(points)

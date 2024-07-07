import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PathCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Example2D would be a structure to hold point data in Python. Assuming it's a namedtuple for simplicity.
from collections import namedtuple
Example2D = namedtuple('Example2D', ['x', 'y', 'label'])

NUM_SHADES = 30

class HeatMap:
    def __init__(self, width, num_samples, x_domain, y_domain, container, show_axes=False, no_svg=False):
        self.num_samples = num_samples
        self.show_axes = show_axes
        self.no_svg = no_svg
        self.x_scale = np.linspace(x_domain[0], x_domain[1], num_samples)
        self.y_scale = np.linspace(y_domain[0], y_domain[1], num_samples)
        
        # Create a color map
        colors = ["#f59322", "#e8eaeb", "#0877bd"]
        self.color_map = LinearSegmentedColormap.from_list("custom", colors, N=NUM_SHADES)
        
        # Setup figure and axes for plotting
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.ax.set_xlim(x_domain)
        self.ax.set_ylim(y_domain)
        
        if not show_axes:
            self.ax.axis('off')
        
        if not no_svg:
            self.train_points = PathCollection([], cmap=self.color_map)
            self.test_points = PathCollection([], cmap=self.color_map)
            self.ax.add_collection(self.train_points)
            self.ax.add_collection(self.test_points)

    def update_test_points(self, points):
        if self.no_svg:
            raise ValueError("Can't add points since no_svg=True")
        self.update_circles(self.test_points, points)

    def update_circles(self, collection, points):
        # Filter points to keep those within the domain
        points = [p for p in points if self.x_scale[0] <= p.x <= self.x_scale[-1] and self.y_scale[0] <= p.y <= self.y_scale[-1]]
        offsets = np.array([(p.x, p.y) for p in points])
        collection.set_offsets(offsets)
        # Assuming 'label' is a value that can be mapped directly to a color
        colors = [p.label for p in points]
        collection.set_array(np.array(colors))
        plt.draw()

    def update_background(self, data, discretize=False):
        if data.shape != (self.num_samples, self.num_samples):
            raise ValueError("Data matrix must be num_samples x num_samples")
        if discretize:
            data = np.where(data >= 0, 1, -1)
        self.ax.imshow(data, extent=(self.x_scale[0], self.x_scale[-1], self.y_scale[0], self.y_scale[-1]), cmap=self.color_map)

def reduce_matrix(matrix, factor):
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square")
    if len(matrix) % factor != 0:
        raise ValueError("Matrix size must be divisible by factor")
    size = len(matrix) // factor
    result = np.zeros((size, size))
    for i in range(0, len(matrix), factor):
        for j in range(0, len(matrix), factor):
            result[i//factor, j//factor] = np.mean(matrix[i:i+factor, j:j+factor])
    return result

# Step 1: Initialize the HeatMap
heatmap = HeatMap(width=10, num_samples=100, x_domain=(-5, 5), y_domain=(-5, 5), container=None, show_axes=True)

# Step 2: Generate Background Data
# Creating a gradient effect for demonstration
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Step 3: Update the HeatMap Background
heatmap.update_background(Z)

# Step 4: Create Test Points
# Creating some test points with labels (color values)
test_points = [
    Example2D(x=-2, y=-2, label=0),  # Label decides the color
    Example2D(x=2, y=2, label=15),
    Example2D(x=0, y=0, label=29)
]

# Step 5: Update Test Points
heatmap.update_test_points(test_points)

# Step 6: Show the Plot
#plt.show()
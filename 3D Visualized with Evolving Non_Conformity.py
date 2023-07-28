import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Create a 3D vector space with 25 examples
data_input = np.random.normal(loc=0, scale=0.1, size=(25, 3))

# Add 25 more examples
data_output = np.random.normal(loc=0.5, scale=0.1, size=(25, 3))

# Combine the data
data = np.vstack([data_input, data_output])

# Function to plot 3D and save figure
def plot_3d(data, data_input, new_conforming_examples, data_output, new_deviating_examples, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(data_input[:, 0], data_input[:, 1], data_input[:, 2], color='turquoise', label='Input Data')
    ax.scatter(data_output[:, 0], data_output[:, 1], data_output[:, 2], color='orange', label='Output Data')
    
    if len(new_conforming_examples) > 0:
        ax.scatter(new_conforming_examples[:, 0], new_conforming_examples[:, 1], new_conforming_examples[:, 2], color='purple', label='New Input Data')
        
    if len(new_deviating_examples) > 0:
        ax.scatter(new_deviating_examples[:, 0], new_deviating_examples[:, 1], new_deviating_examples[:, 2], color='red', label='New Deviating Output Data')
    
    ax.legend()
    ax.set_title('3D visualization of the dataset', fontsize=24)
    plt.savefig(filename)
    plt.show()

# Add 1 conforming and 1 deviating example for each iteration
new_conforming_examples = np.empty((0, 3))
new_deviating_examples = np.empty((0, 3))
for i in range(5):
    new_conforming_example = np.random.normal(loc=0, scale=0.1, size=(1, 3))
    if i < 2:
        new_deviating_example = np.random.normal(loc=1+i*0.1, scale=0.3, size=(1, 3))  # Small deviation for first two iterations
    else:
        new_deviating_example = np.random.normal(loc=1+i*0.5, scale=0.3, size=(1, 3))  # Larger deviation for subsequent iterations
    
    new_conforming_examples = np.vstack([new_conforming_examples, new_conforming_example])
    new_deviating_examples = np.vstack([new_deviating_examples, new_deviating_example])
    
    data_input = np.vstack([data_input, new_conforming_example])
    data_output = np.vstack([data_output, new_deviating_example])
    
    data = np.vstack([data_input, data_output])
    plot_3d(data, data_input, new_conforming_examples, data_output, new_deviating_examples, f'3d_iteration_{i+1}.png')

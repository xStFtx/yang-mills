import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Quaternion Neural Network for Gyroscope Orientation
class QuaternionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QuaternionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Custom loss function to maximize power generation from gyroscopes
def power_loss(output, gravitational_field, n):
    output = output.view(n, 4)
    gravitational_field = gravitational_field.view(n, 4)
    loss = -torch.sum(output * gravitational_field)
    return loss

# Define the dynamic gravitational field using a quaternion representation
def dynamic_gravitational_field(coords, masses, distances, time_step):
    G = 6.67430e-11  # gravitational constant
    grav_field = np.zeros((coords.shape[0], 4))  # Initialize with shape (n, 4)
    for i, coord in enumerate(coords):
        r = np.linalg.norm(coord)
        force_magnitude = sum([G * mass / (r ** 2) for mass in masses])
        direction = np.concatenate((coord / r, [1.0]))  
        grav_field[i] = force_magnitude * direction * np.array([np.cos(time_step), np.sin(time_step), 1, 1])
    return grav_field

# Define the sphere and gyroscope orientations
n = 100  # number of gyroscopes
sphere_coords = np.random.rand(n, 3) * 2 - 1  # random positions in a unit sphere
sphere_coords /= np.linalg.norm(sphere_coords, axis=1, keepdims=True)  # normalize to lie on the sphere

# Initial gyroscope orientations (random quaternions)
gyro_orientations = np.random.rand(n, 4) * 2 - 1
gyro_orientations /= np.linalg.norm(gyro_orientations, axis=1, keepdims=True)

# Define masses and distances for different scales
masses = {
    'earth': 5.972e24,
    'sun': 1.989e30,
    'black_hole': 1.989e40  # large mass for black hole simulation
}
distances = {
    'earth': 6.371e6,
    'sun': 6.957e8,
    'black_hole': 1e4  # close distance for black hole
}

# Select scale to simulate
scale = 'earth'  # change to 'sun' or 'black_hole' for other scales
mass = masses[scale]
distance = distances[scale]

# Initialize the neural network
input_dim = 4  # quaternion
hidden_dim = 64
output_dim = 4  # quaternion
model = QuaternionNN(input_dim, hidden_dim, output_dim)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with dynamic gravitational field
num_epochs = 1000
time_step = 0
for epoch in range(num_epochs):
    model.train()
    
    # Update time step for dynamic field
    time_step += 0.01
    
    # Calculate dynamic gravitational field
    grav_field = dynamic_gravitational_field(sphere_coords, [mass], [distance], time_step)
    grav_field_tensor = torch.tensor(grav_field, dtype=torch.float32)
    
    # Convert to PyTorch tensors
    sphere_coords_tensor = torch.tensor(sphere_coords, dtype=torch.float32)
    gyro_orientations_tensor = torch.tensor(gyro_orientations, dtype=torch.float32)
    
    # Forward pass
    output = model(gyro_orientations_tensor)
    
    # Compute the loss
    loss = power_loss(output, grav_field_tensor, n)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Example of using the trained model
model.eval()
with torch.no_grad():
    example_gyro = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    output_orientation = model(example_gyro)
    print(f"Gyroscope orientation result: {output_orientation.numpy()}")

# Visualization
def plot_universe(black_holes, other_objects):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(black_holes[:, 0], black_holes[:, 1], black_holes[:, 2], c='r', marker='o', label='Black Holes')
    ax.scatter(other_objects[:, 0], other_objects[:, 1], other_objects[:, 2], c='b', marker='^', label='Other Objects')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()
    plt.show()

# Synthetic data for black holes and other large objects
num_black_holes = 50
num_other_objects = 100
black_holes = np.random.rand(num_black_holes, 3) * 1000  # random positions within a 1000 unit cube
other_objects = np.random.rand(num_other_objects, 3) * 1000  # random positions within a 1000 unit cube

# Plot the universe
plot_universe(black_holes, other_objects)

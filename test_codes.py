import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the parameters
Vcruise_values = [5, 10, 15]  # Vcruise: 5 to 15, step 5
RPM_values = [400, 600, 800]  # RPM: 400 to 800, step 200
height_values = [50, 100, 150, 200, 250, 300]  # Height: 50 to 300, step 50

N = 200  # 200x200 grid for noise data

# Function to generate noise data with attenuation
def generate_noise_data(vcruise, rpm, height, N=200):
    """
    Generate a 200x200 grid of noise data with attenuation as you move away from the center.
    The noise increases with higher Vcruise, RPM, and lower height.
    """
    base_noise = (vcruise / max(Vcruise_values)) * 0.5 + (rpm / max(RPM_values)) * 0.3 + (1 - (height / max(height_values))) * 0.2
    noise_grid = np.zeros((N, N))

    center_x, center_y = N // 2, N // 2  # Center of the grid

    # Create a noise pattern centered around the aircraft in the grid
    for i in range(N):
        for j in range(N):
            # Distance from the center
            dist_to_center = np.sqrt((i - center_x)**2 + (j - center_y)**2)

            # Attenuation factor (Inverse-square law attenuation or exponential decay)
            attenuation = 1 / (1 + dist_to_center**2 * 0.0005)  # Inverse-square law

            # Combine base noise with attenuation
            noise_level = base_noise * 70 * attenuation  # Scale noise by attenuation
            noise_grid[i, j] = noise_level + np.random.randn() * 0.5  # Add random noise

    return noise_grid

# Generate the dataset
data = []
for vcruise in Vcruise_values:
    for rpm in RPM_values:
        for height in height_values:
            input_data = [vcruise, rpm, height]
            target_noise = generate_noise_data(vcruise, rpm, height, N)
            data.append((input_data, target_noise))

# Create a Dataset class
class AircraftNoiseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x, y = self.data[index]
        input_data = torch.tensor(x, dtype=torch.float32)
        output_data = torch.tensor(y, dtype=torch.float32)
        return input_data, output_data

# Create the dataset and dataloader
dataset = AircraftNoiseDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Check one batch of data
for inputs, targets in dataloader:
    print("Inputs (Vcruise, RPM, height):", inputs)
    print("Targets (200x200 noise grid):", targets)
    break

# Plot one sample of the noise grid to visualize
example_input, example_target = dataset[0]
plt.imshow(example_target, cmap='hot', interpolation='nearest')
plt.title(f"Noise Grid for Vcruise: {example_input[0]}, RPM: {example_input[1]}, Height: {example_input[2]}")
plt.colorbar(label='Noise dBA')
plt.show()


class NoiseRegressionModel(nn.Module):
    def __init__(self, N):
        super(NoiseRegressionModel, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # Input: Vcruise, RPM, height
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, N * N)  # Output: NxN grid of noise levels

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, N, N)  # Reshape output to NxN grid
        return x


# Initialize the model, loss function, and optimizer
model = NoiseRegressionModel(N)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")


# Test the model
model.eval()
with torch.no_grad():
    test_inputs, test_targets = dataset[0]  # Get a single test sample
    test_inputs = test_inputs.unsqueeze(0)  # Add batch dimension

    predicted_output = model(test_inputs)
    print("Predicted NxN grid:", predicted_output)
    print("Actual NxN grid:", test_targets)

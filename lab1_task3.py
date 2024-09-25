import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

K = 2
# Set random seed for reproducibility
#torch.manual_seed(0)

# synthetic XOR data
def generate_xor_data(n_samples=1000):
    X = np.random.rand(n_samples, 2)
    Y = np.array([1.0 if (y > x + 0.5 or y < x - 0.5) else 0.0 for x, y in X]).reshape(-1, 1)  # Separation with two lines
    return X, Y


class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2, K)  # Two inputs, K neurons in hidden layer
        self.output = nn.Linear(K, 1)   # K outputs from hidden layer to one output neuron
    
    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # Sigmoid activation for hidden layer
        return torch.sigmoid(self.output(x))  # Sigmoid for output layer

# Hyperparameters
goal = 0.01        # Target loss
max_epoch = 10000  # Maximum number of epochs
freq = 100         # Frequency of MSE printing



# X, Y = generate_xor_data(n_samples=1000)
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([0., 1., 1., 0.]).reshape(-1, 1)

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)


model = XORModel()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters())


for epoch in range(max_epoch):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    

    if (epoch + 1) % freq == 0:
        print(f'Epoch [{epoch + 1}/{max_epoch}], Loss: {loss.item():.6f}')


    if loss.item() <= goal:
        print(f'Goal loss reached: {loss.item():.6f} at epoch {epoch + 1}')
        break


xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
with torch.no_grad():
    grid_output = model(grid_tensor).numpy()

# Reshape output back to grid shape for plotting
grid_output = grid_output.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, grid_output, levels=np.linspace(0, 1, 10), cmap='RdYlGn', alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap='RdYlGn', edgecolors='k')
plt.title('XOR Problem - Neural Network Decision Boundary')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.colorbar(label='Output (0 or 1)')
plt.grid(True)
plt.show()


print(f'Final Loss: {loss.item():.6f}')
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
#torch.manual_seed(0)

X = torch.tensor([[3.0], [-7.0]], dtype=torch.float32) 
Y = torch.tensor([[0.25], [0.85]], dtype=torch.float32)

# Uncomment below for test on 3-element vector
# X = torch.tensor([[3.0], [-2.0], [-7.0]], dtype=torch.float32) 
# Y = torch.tensor([[0.25], [0.15], [0.85]], dtype=torch.float32)


class SingleNeuron(nn.Module):
    def __init__(self):
        super(SingleNeuron, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Hyperparameters
goal = 0.005
max_epoch = 1000
freq = 10
learning_rate = 0.05


model = SingleNeuron()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(max_epoch):
    # Reset gradient (PyTorch accumulates gradient by default)
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)
   
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % freq == 0:
        print(f'[{epoch + 1}/{max_epoch}]\t Mean Squared Error: {loss.item():.6f}')

    if loss.item() <= goal:
        print(f'Goal loss reached: {loss.item():.6f} at epoch {epoch + 1}')
        break

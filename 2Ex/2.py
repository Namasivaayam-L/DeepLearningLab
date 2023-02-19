import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)
        self.final = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.final(x)
        return x

# Initialize the model and loss function
model = Net()
criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(10000):
    # Input data
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    # True labels
    y_true = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check the model's prediction
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_pred = model(x)
print(y_pred)

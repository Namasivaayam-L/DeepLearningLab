import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Iris dataset
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

# Split the data into input and target variables
inputs = data.iloc[:, :-1].values
targets = data.iloc[:, -1].values
inputs = inputs
# Standardize the input data
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

# Convert the inputs and targets to PyTorch tensors
inputs = torch.tensor(inputs, dtype=torch.float64)
targets = torch.tensor(targets, dtype=torch.float64)

# Define the RBF network
class RBF(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super(RBF, self).__init__()
        self.hidden_weights = nn.Parameter(torch.randn(num_hidden, num_inputs))
        self.hidden_biases = nn.Parameter(torch.randn(num_hidden))
        self.output_weights = nn.Parameter(torch.randn(num_outputs, num_hidden))
        self.output_biases = nn.Parameter(torch.randn(num_outputs))

    def forward(self, inputs):
        hidden = torch.exp(-torch.sum((inputs[None, :] - self.hidden_weights[:, None, :])**2, dim=-1) / 2)
        hidden = hidden / torch.sum(hidden, dim=0)
        outputs = torch.addmm(self.output_biases, hidden, self.output_weights)
        return outputs

# Initialize the RBF network
rbf = RBF(4, 8, 3)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(rbf.parameters(), lr=0.01)

# Train the RBF network
for epoch in range(100):
    outputs = rbf(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the RBF network on the input data
with torch.no_grad():
    outputs = rbf(inputs)
    _, predictions = torch.max(outputs, dim=1)
    accuracy = torch.mean((predictions == targets).float())
    print("Accuracy:", accuracy)

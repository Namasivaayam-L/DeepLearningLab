import torch

class Perceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor([1, 1]))
        self.bias = torch.nn.Parameter(torch.Tensor([-1.5]))
        
    def bipolar_input(self, x):
        return 2 * x - 1
    
    def bipolar_activation(self, z):
        return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
    
    def forward(self, inputs):
        z = torch.matmul(inputs, self.weights) + self.bias
        return self.bipolar_activation(z)
    
inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
bipolar_inputs = Perceptron.bipolar_input(inputs)
targets = torch.tensor([-1, -1, -1, 1], dtype=torch.float32)

p = Perceptron()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(p.parameters(), lr=0.1)

for epoch in range(100):
    outputs = p(bipolar_inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
for x, t in zip(bipolar_inputs, targets):
    y = p(x)
    print(f"input: {x}, target: {t}, output: {y}")

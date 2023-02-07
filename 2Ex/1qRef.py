import torch
import numpy as np
#f=10+2*x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([12, 14, 16, 18], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def data_generator(data_size=50):
    inputs = []
    labels = []
    for ix in range(data_size):        
        x = np.random.randint(1000) / 1000
        y = 7*(x*x*x) + 8*x+ 2
        inputs.append([x])
        labels.append([y])
        
    return inputs, labels

# model output
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100
X,Y = data_generator()
for epoch in range(n_iters):
    # predict = forward pass
    y_pred = forward(torch.Tensor(X))

    # loss
    l = loss(torch.Tensor(Y), y_pred)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    #w.data = w.data - learning_rate * w.grad
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # zero the gradients after updating
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.8f}')

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')
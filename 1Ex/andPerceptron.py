

import torch
import numpy as np

def perceptron(inputs, weights,bias):
    weighted_sum = torch.dot(inputs, weights)+bias
    prediction = torch.where(weighted_sum > 0, torch.tensor(1.0), torch.tensor(-1.0))
    return prediction

def train_perceptron(inputs, targets, weights, learning_rate, epochs,bias):
    error=0
    for epoch in range(epochs):
        for i, input_ in enumerate(inputs):
            prediction = perceptron(input_, weights,bias)
            error = targets[i] - prediction
            weights += learning_rate * error * input_
        bias+= learning_rate * error
    return weights,bias
def calculate_accuracy(inputs, targets, weights,bias):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, input_ in enumerate(inputs):
            prediction = perceptron(input_, weights,bias)
            if prediction == targets[i]:
                correct += 1
            total += 1
    return correct / total

inputs = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]],dtype=torch.float32)
targets = torch.tensor([-1, -1, -1, 1],dtype=torch.float32)
weights = torch.zeros(2)
# print(weights)
bias = torch.tensor([0.01],dtype=torch.float32)
learning_rate = 0.1
epochs = 20

trained_weights,bias = train_perceptron(inputs, targets, weights, learning_rate, epochs,bias)
print("Trained Weights: ", trained_weights)
accuracy = calculate_accuracy(inputs, targets, trained_weights,bias)
print("Accuracy: ", accuracy*100)
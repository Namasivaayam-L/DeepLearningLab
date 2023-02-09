import numpy as np

# Define the activation function (step function)
def activation_fn(x):
    return 1 if x >= 0 else -1

# Define the prediction function
def predict(inputs, weights):
    summation = np.dot(inputs, weights)
    return activation_fn(summation)

# Define the training function
def train(inputs, targets, weights, learning_rate, epochs):
    for epoch in range(epochs):
        for i, target in enumerate(targets):
            prediction = predict(inputs[i], weights)
            error = target - prediction
            weights += learning_rate * error * inputs[i]
    return weights

# Define the input and target data
inputs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
targets = np.array([1, -1, -1, -1])

# Initialize the weights with random values
weights = np.random.rand(2)

# Set the learning rate and number of epochs
learning_rate = 0.1
epochs = 10

# Train the model
weights = train(inputs, targets, weights, learning_rate, epochs)

# Print the trained weights
print("Trained weights:", weights)

# Make predictions on the input data
for i, target in enumerate(targets):
    prediction = predict(inputs[i], weights)
    print("Input:", inputs[i], "Target:", target, "Prediction:", prediction)

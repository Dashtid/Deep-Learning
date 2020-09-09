import numpy as np


class MLP:
    def __init__(self, x, y, N):
        self.input = x
        self.neuron = N
        self.weights1 = np.random.rand(self.input.shape[1], self.neuron)  # X dimension input connected to N neurons
        self.weights2 = np.random.rand(self.neuron, 1)  # N neurons connected to output
        self.y = y
        self.output = np.zeros(self.y.shape)  # instantiating the output

    #  Method that performs the feed-forward sequence
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    # Method that performs the backpropagation algorithm
    def backpropagation(self):
        # Chain rule to calculate derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T,
                            (2 * (self.y - self.output)
                             * sigmoid_derivative(self.output)))

        d_weights1 = np.dot(self.input.T,
                            (np.dot(2 * (self.y - self.output)
                                    * sigmoid_derivative(self.output),
                                    self.weights2.T) * sigmoid_derivative(self.layer1)))

        # Updating the weights
        self.weights1 += d_weights1
        self.weights2 += d_weights2


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)

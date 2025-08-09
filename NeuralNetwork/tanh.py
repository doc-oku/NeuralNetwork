import numpy as np


class HyperbolicTangent:
    def __init__(self):
        pass

    def forward(self, x):
        x = x.reshape(-1)
        output = np.tanh(x)
        self.grad = 1.0 - output * output
        return output

    def backward(self, x):
        x = x.reshape(-1)
        return x * self.grad

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def adam(self, lr):
        pass
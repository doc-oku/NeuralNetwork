import numpy as np


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        x = x.reshape(-1)
        self.grad = np.where(x > 0.0, 1.0, 0.0)
        return np.maximum(0, x)

    def backward(self, x):
        x = x.reshape(-1)
        return x * self.grad
    
    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass
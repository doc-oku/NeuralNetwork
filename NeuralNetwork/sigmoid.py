import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        x = x.reshape(-1)
        output = 1.0 / (1.0 + np.exp(-x))
        self.grad = output * (1.0 - output)
        return output

    def backward(self, x):
        x = x.reshape(-1)
        back = x * self.grad
        return back

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass

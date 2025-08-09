import numpy as np


class LeakyRelu:
    def __init__(self):
        pass

    def forward(self, x):
        x = x.reshape(-1)
        self.grad = np.where(x > 0.0, 1.0, 0.2)
        output = np.where(x > 0.0, x, 0.2 * x)
        return output

    def backward(self, x):
        x = x.reshape(-1)
        back = x * self.grad
        return back

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def adam(self, lr):
        pass

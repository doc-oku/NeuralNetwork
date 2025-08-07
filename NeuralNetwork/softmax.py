import numpy as np


class SoftMax:
    def __init__(self):
        self.batch_size = None

    def forward(self, x):
        x = x.reshape(self.batch_size, -1)
        maximum = np.max(x, axis=1)
        maximum = maximum.reshape(-1, 1)

        exp_input = np.exp(x - maximum)
        sum1 = np.sum(exp_input, axis=1)
        sum1 = sum1.reshape(-1, 1)

        self.output = exp_input / sum1
        return self.output

    def backward(self, x):
        x = x.reshape(self.batch_size, -1)
        sum1 = np.sum(self.output * x, axis=1)
        sum1 = sum1.reshape(-1, 1)
        back = self.output * (x - sum1)
        return back
    
    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass

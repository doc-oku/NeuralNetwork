import csv
import numpy as np


class Dense:
    def __init__(self):
        self.batch_size = None
        self.input_size = None
        self.output_size = None

    def make(self):
        self.eps = 1e-8
        self.b1 = 0.9
        self.b2 = 0.999
        scale = np.sqrt(2.0 / self.input_size)
        self.weight = np.random.normal(
            loc=0, scale=scale, size=(self.output_size, self.input_size))
        self.d_weight = np.zeros(self.weight.shape)
        self.bias = np.zeros((1, self.output_size))
        self.d_bias = np.zeros((1, self.output_size))
        self.wm = np.zeros((self.output_size, self.input_size))
        self.wv = np.zeros((self.output_size, self.input_size))
        self.bm = np.zeros((1, self.output_size))
        self.bv = np.zeros((1, self.output_size))

    def forward(self, x):
        self.input = x.reshape(self.batch_size, -1)
        output = np.dot(self.input, self.weight.T) + self.bias
        return output

    def backward(self, x):
        self.delta = x.reshape(self.batch_size, -1)
        back = np.dot(self.delta, self.weight)
        return back

    def zero_gradient(self):
        self.d_weight = 0.0
        self.d_bias = 0.0

    def gradient(self):
        self.d_weight += np.dot(self.delta.T, self.input)
        self.d_bias += np.sum(self.delta, axis=0)

    def sgd(self, lr):
        self.d_weight /= self.batch_size
        self.d_bias /= self.batch_size
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias

    def adam(self, lr):
        beta1 = 0.9
        beta2 = 0.999
        self.b1 *= beta1
        self.b2 *= beta2

        self.d_bias /= self.batch_size
        self.bm = beta1 * self.bm + (1.0 - beta1) * self.d_bias
        self.bv = beta2 * self.bv + (1.0 - beta2) * (self.d_bias ** 2.0)
        a = self.bm / (1.0 - self.b1)
        b = self.bv / (1.0 - self.b2)
        self.bias -= lr * a / (np.sqrt(b) + self.eps)

        self.d_weight /= self.batch_size
        self.wm = beta1 * self.wm + (1.0 - beta1) * self.d_weight
        self.wv = beta2 * self.wv + (1.0 - beta2) * (self.d_weight ** 2.0)
        a = self.wm / (1.0 - self.b1)
        b = self.wv / (1.0 - self.b2)
        self.weight -= lr * a / (np.sqrt(b) + self.eps)

    def save(self, folder, path):
        with open(folder+"/"+path+"dsw.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.weight)

        bias = self.bias.reshape(-1, 1)
        with open(folder+"/"+path+"dsb.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(bias)

    def load(self, folder, path):
        with open(folder+"/"+path+"dsw.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for j, row in enumerate(reader):
                for i in range(len(row)):
                    self.weight[j, i] = row[i]

        with open(folder+"/"+path+"dsb.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for j, row in enumerate(reader):
                self.bias[0, j] = row[0]

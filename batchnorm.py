import csv
import numpy as np


class BatchNorm:
    def __init__(self):
        self.batch_size = None
        self.channel = None
        self.map_h = None
        self.map_w = None

    def setting(self):
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8
        map_size = self.map_w * self.map_h
        self.norm = self.batch_size * map_size
        self.gamma = np.ones((self.channel, 1))
        self.beta = np.zeros((self.channel, 1))
        self.d_gamma = np.zeros((self.channel, 1))
        self.d_beta = np.zeros((self.channel, 1))
        self.gm = np.zeros((self.channel, 1))
        self.gv = np.zeros((self.channel, 1))
        self.bm = np.zeros((self.channel, 1))
        self.bv = np.zeros((self.channel, 1))

    def forward(self, x):
        x = x.reshape(self.batch_size, self.channel, -1)
        x = x.transpose(1, 0, 2)
        x = x.reshape(self.channel, -1)
        ave = np.sum(x, axis=1)
        ave /= self.norm
        ave = ave.reshape(-1, 1)

        self.d_input = x - ave
        x = self.d_input*self.d_input
        var = np.sum(x, axis=1)
        var /= self.norm
        var = var.reshape(-1, 1)

        self.buf1 = np.sqrt(var + self.eps)
        self.buf3 = self.buf1 * self.buf1 * self.buf1
        self.xb = self.d_input / self.buf1

        output = self.gamma * self.xb + self.beta
        output = output.reshape(self.channel, self.batch_size, -1)
        output = output.transpose(1, 0, 2)
        return output

    def backward(self, x):
        x = x.reshape(self.batch_size, self.channel, -1)
        x = x.transpose(1, 0, 2)

        self.delta = x.reshape(self.channel, -1)
        d_xb = self.gamma * self.delta
        d_var = np.sum(self.d_input * d_xb, axis=1)
        d_var = d_var.reshape(-1, 1)
        d_var *= -0.5 / self.buf3

        sum1 = np.sum(d_xb, axis=1)
        sum1 = sum1.reshape(-1, 1)
        sum1 /= -self.buf1

        sum2 = np.sum(self.d_input, axis=1)
        sum2 = sum2.reshape(-1, 1)
        sum2 *= -2.0 * d_var / self.norm
        d_ave = sum1 + sum2

        back = d_xb / self.buf1
        back += (2.0 * self.d_input * d_var+d_ave) / self.norm

        back = back.reshape(self.channel, self.batch_size, -1)
        back = back.transpose(1, 0, 2)
        return back

    def zero_gradient(self):
        self.d_gamma = 0.0
        self.d_beta = 0.0

    def gradient(self):
        self.d_gamma += np.sum(self.delta * self.xb, axis=1).reshape(-1, 1)
        self.d_beta += np.sum(self.delta, axis=1).reshape(-1, 1)

    def sgd(self, lr):
        self.d_gamma /= self.batch_size
        self.gamma -= lr * self.d_gamma
        self.d_beta /= self.batch_size
        self.beta -= lr * self.d_beta

    def adam(self, lr):
        beta1 = 0.9
        beta2 = 0.999
        self.b1 *= beta1
        self.b2 *= beta2

        self.d_gamma /= self.batch_size
        self.gm = beta1 * self.gm + (1.0 - beta1) * self.d_gamma
        self.gv = beta2 * self.gv + (1.0 - beta2) * self.d_gamma * self.d_gamma
        m = self.gm / (1.0 - self.b1)
        v = self.gv / (1.0 - self.b2)
        self.gamma -= lr * m / (np.sqrt(v) + self.eps)

        self.d_beta /= self.batch_size
        self.bm = beta1 * self.bm + (1.0 - beta1) * self.d_beta
        self.bv = beta2 * self.bv + (1.0 - beta2) * self.d_beta * self.d_beta
        m = self.bm / (1.0 - self.b1)
        v = self.bv / (1.0 - self.b2)
        self.beta -= lr * m / (np.sqrt(v) + self.eps)

    def save(self, folder, path):
        with open(folder+"/" + path + "nmg.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.gamma)

        with open(folder+"/" + path + "nmb.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.beta)

    def load(self, folder, path):
        with open(folder+"/" + path + "nmg.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for j, row in enumerate(reader):
                self.gamma[j, 0] = row[0]

        with open(folder+"/" + path + "nmb.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for j, row in enumerate(reader):
                self.beta[j, 0] = row[0]

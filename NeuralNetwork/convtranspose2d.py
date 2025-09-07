import csv
import numpy as np


class ConvTranspose2d:
    def __init__(self):
        self.stride = None
        self.batch_size = None
        self.output_channel = None
        self.input_channel = None
        self.input_h = None
        self.input_w = None
        self.filter_w = None
        self.filter_h = None

    def make(self):
        self.eps = 1e-8
        self.b1 = 0.9
        self.b2 = 0.999
        self.weight = np.random.normal(loc=0, scale=0.02, size=(
            self.output_channel, self.input_channel, self.filter_h, self.filter_w))
        self.d_weight = np.zeros(self.weight.shape)
        self.wm = np.zeros(self.weight.shape)
        self.wv = np.zeros(self.weight.shape)
        self.output_h = int(self.input_h * self.stride)
        self.output_w = int(self.input_w * self.stride)

        if self.pad == True:
            self.pad_h = int(self.filter_h / 2)
            self.pad_w = int(self.filter_w / 2)
        else:
            self.pad_h = 0
            self.pad_w = 0

    def col2img(self, col):
        col = col.reshape(self.batch_size, self.output_h, self.output_w,
                          self.input_channel, self.filter_h, self.filter_w)

        pad_col = np.pad(col, [(0, 0), (self.pad_h, self.pad_h),
                               (self.pad_w, self.pad_w), (0, 0), (0, 0), (0, 0)], 'constant')

        pad_col = pad_col.transpose(0, 3, 4, 5, 1, 2)  # b,ic,fh,fw,oh,ow
        img = np.zeros((self.batch_size, self.input_channel,
                       self.input_h, self.input_w))

        for j in range(self.filter_h):
            j_max = j + self.output_h
            for i in range(self.filter_w):
                i_max = i + self.output_w
                img[:, :, :, :] += pad_col[:, :, j, i,
                                           j:j_max:self.stride, i:i_max:self.stride]
        return img

    def img2col(self, img):
        output_h = int(self.output_h + 2 * self.pad_h)
        output_w = int(self.output_w + 2 * self.pad_w)
        col = np.zeros(
            (self.batch_size, self.input_channel, self.filter_h, self.filter_w, output_h, output_w))

        for j in range(self.filter_h):
            j_max = j + self.output_h
            for i in range(self.filter_w):
                i_max = i + self.output_w
                col[:, :, j, i, j:j_max:self.stride,
                    i:i_max:self.stride] = img[:, :, :, :]

        col = col[:, :, :, :, self.pad_h: self.output_h +
                  self.pad_h, self.pad_w: self.output_w + self.pad_w]
        col = col.transpose(0, 4, 5, 1, 2, 3)
        col = col.reshape(self.batch_size * self.output_h * self.output_w, -1)
        return col

    def forward(self, x):
        img = x.reshape(self.batch_size, self.input_channel, self.input_h, -1)
        self.col = self.img2col(img)
        self.col_w = self.weight.reshape(self.output_channel, -1)
        output = np.dot(self.col, self.col_w.T)
        output = output.reshape(
            self.batch_size, self.output_h, self.output_w, -1)
        output = output.transpose(0, 3, 1, 2)
        return output

    def backward(self, x):
        self.delta = x.reshape(
            self.batch_size, self.output_channel, self.output_h, -1)
        x = self.delta.transpose(0, 2, 3, 1)
        x = x.reshape(-1, self.output_channel)
        col = np.dot(x, self.col_w)
        back = self.col2img(col)
        return back

    def zero_gradient(self):
        self.d_weight = 0.0

    def gradient(self):
        x = self.delta.transpose(1, 0, 2, 3)
        x = x.reshape(self.output_channel, -1)
        self.d_weight += np.dot(x, self.col).reshape(self.weight.shape)

    def sgd(self, lr):
        self.d_weight /= self.batch_size
        self.weight -= lr * self.d_weight

    def adam(self, lr):
        beta1 = 0.9
        beta2 = 0.999
        self.b1 *= beta1
        self.b2 *= beta2
        self.d_weight /= self.batch_size
        self.wm = beta1 * self.wm + (1.0 - beta1) * self.d_weight
        self.wv = beta2 * self.wv + (1.0 - beta2) * (self.d_weight ** 2)
        a = self.wm / (1.0 - self.b1)
        b = self.wv / (1.0 - self.b2)
        self.weight -= lr * a / (np.sqrt(b) + self.eps)

    def save(self, folder, path):
        weight = self.weight.reshape(
            self.output_channel*self.input_channel, -1)
        with open(folder+"/"+path+"cvw.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(weight)

    def load(self, folder, path):
        self.weight = self.weight.reshape(
            self.output_channel*self.input_channel, -1)
        with open(folder+"/"+path+"cvw.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for j, row in enumerate(reader):
                for i in range(len(row)):
                    self.weight[j, i] = row[i]
        self.weight = self.weight.reshape(
            self.output_channel, self.input_channel, self.filter_h, -1)

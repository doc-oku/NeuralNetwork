import numpy as np


class Pooling:
    def __init__(self):
        self.kernel_w = None
        self.kernel_h = None
        self.batch_size = None
        self.channel = None
        self.input_w = None
        self.input_h = None
        self.output_h = None
        self.output_w = None

    def setting(self):
        self.kernel_size = self.kernel_w * self.kernel_h
        self.output_h = int((self.input_h - self.kernel_h) / self.kernel_h + 1)
        self.output_w = int((self.input_w - self.kernel_w) / self.kernel_h + 1)

    def col2img(self, x):
        col = x.reshape(self.batch_size, self.output_h, self.output_w,
                        self.channel, self.kernel_h, self.kernel_w)
        col = col.transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((self.batch_size, self.channel,
                       self.input_h, self.input_w))

        for j in range(self.kernel_h):
            j_max = j + self.kernel_h * self.output_h
            for i in range(self.kernel_w):
                i_max = i + self.kernel_w * self.output_w
                img[:, :, j:j_max:self.kernel_h,
                    i:i_max:self.kernel_w] += col[:, :, j, i, :, :]
        return img

    def img2col(self, img):
        col = np.zeros(
            (self.batch_size, self.channel, self.kernel_h, self.kernel_w, self.output_h, self.output_w))

        for j in range(self.kernel_h):
            j_max = j + self.kernel_h * self.output_h
            for i in range(self.kernel_w):
                i_max = i + self.kernel_w * self.output_w
                col[:, :, j, i, :, :] = img[:, :,
                                            j:j_max:self.kernel_h, i:i_max:self.kernel_w]

        col = col.transpose(0, 4, 5, 1, 2, 3)
        col = col.reshape(self.batch_size * self.output_h * self.output_w, -1)
        return col

    def forward(self, x):
        img = x.reshape(self.batch_size, self.channel, self.input_h, -1)
        col = self.img2col(img)
        col = col.reshape(-1, self.kernel_h * self.kernel_w)
        self.arg_max = np.argmax(col, axis=1)
        output = np.max(col, axis=1)
        output = output.reshape(
            self.batch_size, self.output_h, self.output_w, -1)
        output = output.transpose(0, 3, 1, 2)
        return output

    def backward(self, x):
        delta = x.reshape(self.batch_size, self.channel, self.output_h, -1)
        delta = delta.transpose(0, 2, 3, 1)
        col = np.zeros((delta.size, self.kernel_size))
        col[np.arange(self.arg_max.size),
            self.arg_max.flatten()] = delta.flatten()
        back = self.col2img(col)
        return back

    def zero_gradient(self):
        pass

    def gradient(self):
        pass

    def sgd(self, lr):
        pass

    def adam(self, lr):
        pass

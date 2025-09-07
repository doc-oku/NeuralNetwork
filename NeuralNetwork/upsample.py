import cupy as cp


class Upsample:
    def __init__(self):
        self.batch_size = None
        self.channel = None
        self.input_w = None
        self.input_h = None

    def make(self):
        self.output_h = int(self.input_h * self.kernel_h)
        self.output_w = int(self.input_w * self.kernel_w)
        self.kernel_size = self.kernel_w * self.kernel_h

    def col2img(self, col):
        col = col.reshape(self.batch_size, self.output_h, self.output_w,
                          self.channel, self.kernel_h, self.kernel_w)
        col = col.transpose(0, 3, 4, 5, 1, 2)
        img = cp.zeros((self.batch_size, self.channel,
                       self.input_h, self.input_w))

        for j in range(self.kernel_h):
            j_max = j + self.output_h
            for i in range(self.kernel_w):
                i_max = i + self.output_w
                img[:, :, :, :] += col[:, :, j, i,
                                       j:j_max:self.kernel_h, i:i_max:self.kernel_w]
        return img

    def img2col(self, img):
        col = cp.zeros(
            (self.batch_size, self.channel, self.kernel_h, self.kernel_w, self.output_h, self.output_w))

        for j in range(self.kernel_h):
            j_max = j + self.output_h
            for i in range(self.kernel_w):
                i_max = i + self.output_w
                col[:, :, j, i, j:j_max:self.kernel_h,
                    i:i_max:self.kernel_w] = img[:, :, :, :]

        col = col.transpose(0, 4, 5, 1, 2, 3)
        col = col.reshape(self.batch_size * self.output_h * self.output_w, -1)
        return col

    def forward(self, x):
        img = x.reshape(self.batch_size, self.channel, self.input_h, -1)
        col = self.img2col(img)
        col = col.reshape(-1, self.kernel_size)
        x = cp.max(col, axis=1)
        x = x.reshape(self.batch_size, self.output_h, self.output_w, -1)
        output = x.transpose(0, 3, 1, 2)
        return output

    def backward(self, x):
        x = x.reshape(self.batch_size, self.channel, self.output_h, -1)
        x = x.transpose(0, 2, 3, 1)
        x = x.flatten()
        col = cp.broadcast_to(x, (self.kernel_size, x.size))
        col = col.transpose(1, 0)
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

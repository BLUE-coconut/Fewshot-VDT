import mindspore as ms
import mindspore.numpy as mnp
import mindspore.nn as nn
from mindspore import Tensor, ops
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss_np = np.array([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                         for x in range(window_size)], dtype=np.float32)
    gauss_np = gauss_np / gauss_np.sum()
    return Tensor(gauss_np, ms.float32)


def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).reshape(window_size, 1)
    _2D = ops.matmul(_1D, _1D.T)
    _2D = _2D.reshape(1, 1, window_size, window_size)

    window = ops.tile(_2D, (channel, 1, 1, 1))
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    conv2d = ops.Conv2D(out_channel=channel,
                        kernel_size=window_size,
                        pad_mode='pad',
                        pad=window_size // 2,
                        group=channel)

    mu1 = conv2d(img1, window)
    mu2 = conv2d(img2, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window) - mu2_sq
    sigma12 = conv2d(img1 * img2, window) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(axis=(1, 2, 3))


class SSIM(nn.Cell):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def construct(self, img1, img2):
        _, channel, _, _ = img1.shape

        if channel != self.channel:
            self.window = create_window(self.window_size, channel)
            self.channel = channel

        window = self.window.astype(img1.dtype)

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    _, channel, _, _ = img1.shape
    window = create_window(window_size, channel).astype(img1.dtype)
    return _ssim(img1, img2, window, window_size, channel, size_average)

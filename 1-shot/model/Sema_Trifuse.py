import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import functional as F

class ChannelAttention(nn.Cell):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.SequentialCell([
            nn.Dense(channel, channel // 4, has_bias=False),
            nn.ReLU()
        ])
        self.fc2 = nn.SequentialCell([
            nn.Dense(channel // 4, channel, has_bias=False),
            nn.Sigmoid()
        ])
        self.reshape = ops.Reshape()
        self.expand = ops.BroadcastTo
        self.mul = ops.Mul()
        self.add = ops.Add()

    def construct(self, x):
        n, c, h, w = x.shape
        y1 = self.avg_pool(x)
        y1 = self.reshape(y1, (n, -1))
        y = self.fc2(self.fc1(y1))
        y = self.reshape(y, (n, c, 1, 1))
        y = y.broadcast_to(x.shape)
        y = self.mul(x, y)
        y = self.add(y, x)
        return y


class SpatialAttention(nn.Cell):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 192, kernel_size=3, pad_mode='pad', padding=1, has_bias=False)
        self.sigmoid = nn.Sigmoid()
        self.cat = ops.Concat(axis=1)
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.reduce_mean = ops.ReduceMean(keep_dims=True)
        self.reduce_max = ops.ReduceMax(keep_dims=True)

    def construct(self, x):
        avg_out = self.reduce_mean(x, 1)
        max_out = self.reduce_max(x, 1)[0]
        scale = self.cat((avg_out, max_out))
        scale = self.conv(scale)
        out = self.mul(x, self.sigmoid(scale))
        out = self.add(out, x)
        return out


class TriFusenet(nn.Cell):
    def __init__(self, in_dim=64, middle_dim=64, out_dim=64, add_channels=384, end_channels=128, kernel_size=2):
        super(TriFusenet, self).__init__()
        self.start_convs = nn.CellList()
        self.filter_convs = nn.CellList()
        self.gate_convs = nn.CellList()
        self.residual_convs = nn.CellList()
        self.bn = nn.CellList()
        self.Sa = SpatialAttention()
        self.Ca = ChannelAttention(192)
        self.concat0 = ops.Concat(axis=0)
        self.concat1 = ops.Concat(axis=1)
        self.tanh = ops.Tanh()
        self.sigmoid = ops.Sigmoid()
        self.relu = ops.ReLU()

        for i in range(3):
            self.start_convs.append(nn.Conv2d(in_channels=in_dim, out_channels=middle_dim, kernel_size=1))
            self.filter_convs.append(nn.Conv2d(in_channels=middle_dim, out_channels=out_dim, kernel_size=(1, 3), pad_mode='pad', padding=(0, 1)))
            self.gate_convs.append(nn.Conv1d(in_channels=middle_dim, out_channels=out_dim, kernel_size=3, pad_mode='pad', padding=1))
            self.residual_convs.append(nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1))
            self.bn.append(nn.BatchNorm2d(out_dim))

        self.end_conv_1 = nn.Conv2d(add_channels, end_channels, kernel_size=1, has_bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=1, has_bias=True)

    def construct(self, rgb, th, d):
        expand = ops.ExpandDims()
        rgb = expand(rgb, 0)
        th = expand(th, 0)
        d = expand(d, 0)
        tri = self.concat0((rgb, th, d))

        enh = []
        for i in range(3):
            sin = tri[i]
            ori = sin
            sin = self.start_convs[i](sin)
            mix_1 = self.tanh(self.filter_convs[i](sin))
            mix_2 = self.sigmoid(self.gate_convs[i](sin))
            mix = mix_1 * mix_2
            ori = self.residual_convs[i](ori)
            add = self.bn[i](ori + mix)
            enh.append(add)

        tri_sc = self.concat1((enh[0], enh[1], enh[2]))
        tri_sa_1 = self.Sa(tri_sc)
        tri_ca_1 = self.Ca(tri_sc)
        tri_2 = self.relu(self.concat1((tri_sa_1, tri_ca_1)))
        tri_3 = self.relu(self.end_conv_1(tri_2))
        tri_final = self.end_conv_2(tri_3)
        return tri_final

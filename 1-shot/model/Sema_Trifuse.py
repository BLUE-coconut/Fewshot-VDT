import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Linear(channel, channel // 4, bias=False), nn.ReLU(inplace=True))
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Sequential(nn.Linear(channel // 4, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        n, c, h, w = x.shape        # 2 192 100 100
        y1 = self.avg_pool(x)       # 2 192 1   1
        y1 = y1.reshape(n, -1)      # 2 192
        y = self.fc2(self.fc1(y1))  # 2 192
        y = y.reshape(n, c, 1, 1).expand_as(x).clone()  # 2 192 100 100
        y = x * y + x
        return y


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 192, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        out = x * self.sigmoid(scale) + x

        return out


class TriFusenet(nn.Module):
    def __init__(self, in_dim=64, middle_dim=64, out_dim=64, add_channels=384, end_channels=128, kernel_size=2):
        super(TriFusenet, self).__init__()
        self.start_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.Sa = SpatialAttention()
        self.Ca = ChannelAttention(192)

        for i in range(3):
            self.start_convs.append(nn.Conv2d(in_channels=in_dim,
                                              out_channels=middle_dim,
                                              kernel_size=(1, 1)))
            # dilated convolutions
            self.filter_convs.append(nn.Conv2d(in_channels=middle_dim,
                                               out_channels=out_dim,
                                               kernel_size=(1, 3),
                                               padding=(0, 1)))

            self.gate_convs.append(nn.Conv1d(in_channels=middle_dim,
                                             out_channels=out_dim,
                                             kernel_size=(1, 3),
                                             padding=(0, 1)))
            # 1x1 convolution for residual connection
            self.residual_convs.append(nn.Conv1d(in_channels=in_dim,
                                                 out_channels=out_dim,
                                                 kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(out_dim))

        self.end_conv_1 = nn.Conv2d(in_channels=add_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, rgb, th, d):
        rgb = rgb.unsqueeze(0)              # 1 2 64 100 100
        th = th.unsqueeze(0)
        d = d.unsqueeze(0)
        tri = torch.cat((rgb, th, d), 0)  # 3 2 64 100 100

        enh = []

        for i in range(3):
            sin = tri[i]
            ori = sin
            sin = self.start_convs[i](sin)

            mix_1 = self.filter_convs[i](sin)  # 1 64 100 100
            mix_1 = torch.tanh(mix_1)
            mix_2 = self.gate_convs[i](sin)
            mix_2 = torch.sigmoid(mix_2)
            mix = mix_1 * mix_2

            ori = self.residual_convs[i](ori)
            add = ori + mix
            add = self.bn[i](add)
            enh.append(add)

        tri_sc = torch.cat((enh[0], enh[1], enh[2]), 1)  # 2 192 100 100

        tri_sa_1 = self.Sa(tri_sc)
        tri_ca_1 = self.Ca(tri_sc)

        tri_2 = torch.cat((tri_sa_1, tri_ca_1), 1)  # 2 384 100 100

        tri_2 = F.relu(tri_2)
        tri_3 = F.relu(self.end_conv_1(tri_2))
        tri_final = self.end_conv_2(tri_3)
        return tri_final

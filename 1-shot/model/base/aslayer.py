import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor
from einops import rearrange


class AttentiveSqueezeLayer(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, heads=8, groups=4, pool_kv=False):
        super(AttentiveSqueezeLayer, self).__init__()
        self.attn = Attention(in_channels, out_channels, kernel_size, stride, padding, bias, heads, groups, pool_kv)
        self.ff = FeedForward(out_channels, groups)

    def construct(self, input):
        x, support_mask = input
        batch, c, qh, qw, sh, sw = x.shape
        x = rearrange(x, 'b c d t h w -> b c (d t) h w')
        out = self.attn((x, support_mask))
        out = self.ff(out)
        out = rearrange(out, 'b c (d t) h w -> b c d t h w', d=qh, t=qw)
        return out, support_mask


class Attention(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, heads=8, groups=4, pool_kv=False):
        super(Attention, self).__init__()
        self.heads = heads

        self.softmax = ops.Softmax(axis=-1)
        self.einsum = ops.Einsum('b g c t l, b g c t m -> b g t l m', 'b g t l m, b g c t m -> b g c t l')
        self.masked_fill_value = Tensor(-1e9, mindspore.float32)

        retain_dim = in_channels == out_channels and math.floor((2 * padding - kernel_size) / stride) == -1
        hidden_channels = out_channels // 2
        assert hidden_channels % self.heads == 0, "out_channels should be divided by heads. (example: out_channels: 40, heads: 4)"

        ksz_q = (1, kernel_size, kernel_size)
        str_q = (1, stride, stride)
        pad_q = (0, padding, padding)

        self.short_cut = nn.SequentialCell(
            nn.Conv3d(in_channels, out_channels, kernel_size=ksz_q, stride=str_q, padding=pad_q, has_bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU()
        ) if not retain_dim else nn.Identity()

        self.qhead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz_q, stride=str_q, padding=pad_q, has_bias=bias)

        ksz = (1, kernel_size, kernel_size) if pool_kv else (1, 1, 1)
        str = (1, stride, stride) if pool_kv else (1, 1, 1)
        pad = (0, padding, padding) if pool_kv else (0, 0, 0)

        self.khead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz, stride=str, padding=pad, has_bias=bias)
        self.vhead = nn.Conv3d(in_channels, hidden_channels, kernel_size=ksz, stride=str, padding=pad, has_bias=bias)

        self.agg = nn.SequentialCell([
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, has_bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU()
        ])
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def construct(self, input):
        x, support_mask = input

        x_ = self.short_cut(x)
        q_out = self.qhead(x)
        k_out = self.khead(x)
        v_out = self.vhead(x)

        q_h, q_w = q_out.shape[-2:]
        k_h, k_w = k_out.shape[-2:]

        q_out = rearrange(q_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)
        k_out = rearrange(k_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)
        v_out = rearrange(v_out, 'b (g c) t h w -> b g c t (h w)', g=self.heads)

        out = ops.einsum('b g c t l, b g c t m -> b g t l m', q_out, k_out)
        out = self.attn_mask(out, support_mask, spatial_size=(k_h, k_w))

        out = self.softmax(out)

        out = ops.einsum('b g t l m, b g c t m -> b g c t l', out, v_out)
        
        out = rearrange(out, 'b g c t (h w) -> b (g c) t h w', h=q_h, w=q_w)
        out = self.agg(out)

        return self.out_norm(out + x_)

    def attn_mask(self, x, mask, spatial_size):

        mask_float = mask.astype(mindspore.float32).expand_dims(1)
        mask_interp = ops.interpolate(mask_float, sizes=spatial_size, coordinate_transformation_mode="align_corners", mode="bilinear")
        
        mask_interp = rearrange(mask_interp, 'b 1 h w -> b 1 1 1 (h w)')

        condition = mask_interp == 0
        out = ops.select(condition, self.masked_fill_value, x)
        return out


class FeedForward(nn.Cell):
    def __init__(self, out_channels, groups=4, size=2):
        super(FeedForward, self).__init__()
        hidden_channels = out_channels // size
        self.ff = nn.SequentialCell([
            nn.Conv3d(out_channels, hidden_channels, kernel_size=1, stride=1, padding=0, has_bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, has_bias=False),
        ])
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def construct(self, x):
        x_ = x
        out = self.ff(x)
        return self.out_norm(out + x_)
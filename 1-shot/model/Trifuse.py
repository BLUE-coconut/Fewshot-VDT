import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from .base.transformer import MultiHeadedAttention
from .base.correlation import Correlation, cosine_similarity

def ConvModule(in_ch, out_ch, kernel, padding=None, mode="sigmoid"):
    if padding is None:
        padding = kernel // 2
    if mode == "sigmoid":
        conv = nn.SequentialCell([
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, pad_mode='pad', padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.Sigmoid()
        ])
    elif mode == "relu":
        conv = nn.SequentialCell([
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, pad_mode='pad', padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        ])
    elif mode == "bn":
        conv = nn.SequentialCell([
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, pad_mode='pad', padding=padding),
            nn.BatchNorm2d(out_ch)
        ])
    else:
        conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, pad_mode='pad', padding=padding)
    return conv


class TriCorr_v2_6(nn.Cell):
    def __init__(self, in_ch, mid_ch=None, out_ch=None, dropout=0.5):
        super(TriCorr_v2_6, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch if mid_ch is not None else in_ch
        self.out_ch = out_ch if out_ch is not None else self.mid_ch

        self.conv_DT = ConvModule(in_ch=self.in_ch*2, out_ch=self.mid_ch, kernel=1)
        self.reduct_V = ConvModule(in_ch=self.in_ch, out_ch=self.mid_ch, kernel=1)
        self.MHA_V = MultiHeadedAttention(num_heads=8, d_model=self.mid_ch, dropout=dropout)

        self.Q_sim_conv = ConvModule(1, 1, 1, mode="conv")
        self.S_sim_conv = ConvModule(1, 1, 1, mode="conv")

        self.V_corr = ConvModule(self.mid_ch*2, self.out_ch, 3)
        self.concat = ops.Concat(axis=1)
        self.mul = ops.Mul()

    def construct(self, query, support):
        QV, QD, QT = query
        SV, SD, ST = support

        fused_Trimaps = []

        QV = self.reduct_V(QV)
        SV = self.reduct_V(SV)

        sim = cosine_similarity(QV, SV)

        REQ = self.mul(self.Q_sim_conv(sim), QV)
        Q_en_V = self.MHA_V(self.conv_DT(self.concat([QD, QT])), REQ, REQ)

        REV = self.mul(self.S_sim_conv(sim), SV)
        S_en_V = self.MHA_V(self.conv_DT(self.concat([SD, ST])), REV, REV)

        fused_Trimaps.append(self.V_corr(self.concat([Q_en_V, S_en_V])))

        return fused_Trimaps

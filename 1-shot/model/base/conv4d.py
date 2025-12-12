import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class CenterPivotConv4d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               has_bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               has_bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

        self.arange = ops.Arange()
        self.sum = ops.ReduceSum()

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.shape
        if not self.idx_initialized:
            idxh = self.arange(Tensor(0, mindspore.int32), Tensor(hb, mindspore.int32), Tensor(self.stride[2:][0], mindspore.int32))
            idxw = self.arange(Tensor(0, mindspore.int32), Tensor(wb, mindspore.int32), Tensor(self.stride[2:][1], mindspore.int32))
            
            self.len_h = idxh.shape[0]
            self.len_w = idxw.shape[0]

            idxw_tiled = ops.tile(idxw.expand_dims(0), (self.len_h, 1))
            idxh_tiled = ops.tile(idxh.expand_dims(0), (self.len_w, 1)).transpose()
            
            self.idx = (idxw_tiled + idxh_tiled * wb).reshape(-1).astype(mindspore.int32)
            self.idx_initialized = True

        ct_reshaped = ct.reshape(bsz, ch, ha, wa, -1)
        

        ct_pruned = ct_reshaped.reshape(-1, ct_reshaped.shape[-1])
        ct_pruned = ops.gather(ct_pruned, 1, self.idx)
        ct_pruned = ct_pruned.reshape(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def construct(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
            
        bsz, inch, ha, wa, hb, wb = out1.shape

        out1 = out1.transpose((0, 4, 5, 1, 2, 3)).reshape((-1, inch, ha, wa))
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.shape[-3:]

        out1 = out1.reshape((bsz, hb, wb, outch, o_ha, o_wa)).transpose((0, 3, 4, 5, 1, 2))

        bsz, inch, ha, wa, hb, wb = x.shape

        out2 = x.transpose((0, 2, 3, 1, 4, 5)).reshape((-1, inch, hb, wb))
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.shape[-3:]

        out2 = out2.reshape((bsz, ha, wa, outch, o_hb, o_wb)).transpose((0, 3, 1, 2, 4, 5))

        if out1.shape[-2:] != out2.shape[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.reshape((bsz, outch, o_ha, o_wa, -1))
            out1 = self.sum(out1, axis=-1)

            out2 = ops.squeeze(out2)

        y = out1 + out2
        return y

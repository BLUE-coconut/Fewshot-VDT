from functools import reduce
from operator import add

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
# Note: MindSpore uses nn.Cell instead of nn.Module
# torch.nn.functional (F) operations like interpolate are replaced by ops.ResizeBilinearV2

# Assuming the custom modules are available as MindSpore nn.Cell implementations
from .base.aslayer import AttentiveSqueezeLayer
from .base.conv4d import CenterPivotConv4d as Conv4d


class HPNLearner(nn.Cell):

    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        self.interpolate = ops.ResizeBilinearV2(half_pixel_centers=False)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.mean = ops.ReduceMean(keep_dims=False)

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4

                str4d = (1, 1) + (stride,) * 2 
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU())

            return nn.SequentialCell(*building_block_layers)

        outch1, outch2, outch3 = 16, 64, 128

        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        self.decoder1 = nn.SequentialCell(
            nn.Conv2d(outch3, outch3, (3, 3), pad_mode='pad', padding=(1, 1), has_bias=True),
            nn.ReLU(),
            nn.Conv2d(outch3, outch2, (3, 3), pad_mode='pad', padding=(1, 1), has_bias=True),
            nn.ReLU()
        )

        self.decoder2 = nn.SequentialCell(
            nn.Conv2d(outch2, outch2, (3, 3), pad_mode='pad', padding=(1, 1), has_bias=True),
            nn.ReLU(),
            nn.Conv2d(outch2, 2, (3, 3), pad_mode='pad', padding=(1, 1), has_bias=True)
        )

    def interpolate_support_dims(self, hypercorr, spatial_size):

        bsz, ch, ha, wa, hb, wb = hypercorr.shape
        hypercorr = self.transpose(hypercorr, (0, 4, 5, 1, 2, 3))
        hypercorr = self.reshape(hypercorr, (bsz * hb * wb, ch, ha, wa))

        o_hb, o_wb = spatial_size
        hypercorr = self.interpolate(hypercorr, size=(o_hb, o_wb))
        hypercorr = self.reshape(hypercorr, (bsz, hb, wb, ch, o_hb, o_wb))
        hypercorr = self.transpose(hypercorr, (0, 3, 4, 5, 1, 2))
        return hypercorr

    def construct(self, hypercorr_pyramid):

        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        target_size_3 = hypercorr_sqz3.shape[-4:-2]
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, target_size_3)
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        target_size_2 = hypercorr_sqz2.shape[-4:-2]
        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, target_size_2)
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.shape
        hypercorr_encoded = self.reshape(hypercorr_mix432, (bsz, ch, ha, wa, -1))
        hypercorr_encoded = self.mean(hypercorr_encoded, 4)
        hypercorr_decoded = self.decoder1(hypercorr_encoded)

        upsample_size = (hypercorr_decoded.shape[-1] * 2, hypercorr_decoded.shape[-2] * 2) 
        hypercorr_decoded = self.interpolate(hypercorr_decoded, upsample_size)
        
        logit_mask = self.decoder2(hypercorr_decoded)

        return logit_mask


class AttentionLearner(nn.Cell):
    """
    Attentive Squeeze Hypercorrelation Pyramid Network Learner (AS-HPN)
    Processes 4D hypercorrelations using Attentive Squeeze Layers.
    """
    def __init__(self, inch):
        super(AttentionLearner, self).__init__()

        self.interpolate = ops.ResizeBilinearV2(half_pixel_centers=False)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.mean_keep_dims = ops.ReduceMean(keep_dims=True)
        self.mean = ops.ReduceMean(keep_dims=False)

        def make_building_attentive_block(in_channel, out_channels, kernel_sizes, spt_strides, pool_kv=False):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)
            
            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                padding = ksz // 2 if ksz > 2 else 0

                building_block_layers.append(AttentiveSqueezeLayer(inch, outch, ksz, stride, padding, pool_kv=pool_kv))

            return nn.SequentialCell(*building_block_layers)

        self.feat_ids = list(range(4, 17))

        self.encoder_layer4 = make_building_attentive_block(inch[0], [32, 128], [5, 3], [4, 2])
        self.encoder_layer3 = make_building_attentive_block(inch[1], [32, 128], [5, 5], [4, 4], pool_kv=True)
        self.encoder_layer2 = make_building_attentive_block(inch[2], [32, 128], [5, 5], [4, 4], pool_kv=True)

        self.encoder_layer4to3 = make_building_attentive_block(128, [128, 128], [1, 2], [1, 1])
        self.encoder_layer3to2 = make_building_attentive_block(128, [128, 128], [1, 2], [1, 1])

        # Decoder layers
        self.decoder1 = nn.SequentialCell(
            nn.Conv2d(128, 128, (3, 3), pad_mode='pad', padding=(1, 1), has_bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, (3, 3), pad_mode='pad', padding=(1, 1), has_bias=True),
            nn.ReLU()
        )

        self.decoder2 = nn.SequentialCell(
            nn.Conv2d(64, 64, (3, 3), pad_mode='pad', padding=(1, 1), has_bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 2, (3, 3), pad_mode='pad', padding=(1, 1), has_bias=True)
        )

    def interpolate_query_dims(self, hypercorr, spatial_size):
        bsz, ch, ha, wa, hb, wb = hypercorr.shape

        hypercorr = self.transpose(hypercorr, (0, 4, 5, 1, 2, 3))
        hypercorr = self.reshape(hypercorr, (bsz * hb * wb, ch, ha, wa))

        o_ha, o_wa = spatial_size
        hypercorr = self.interpolate(hypercorr, size=(o_ha, o_wa))

        hypercorr = self.reshape(hypercorr, (bsz, hb, wb, ch, o_ha, o_wa))
        hypercorr = self.transpose(hypercorr, (0, 3, 4, 5, 1, 2))
        return hypercorr

    def construct(self, hypercorr_pyramid, support_mask):

        hypercorr_sqz4, _ = self.encoder_layer4((hypercorr_pyramid[0], support_mask))
        hypercorr_sqz3, _ = self.encoder_layer3((hypercorr_pyramid[1], support_mask))
        hypercorr_sqz2, _ = self.encoder_layer2((hypercorr_pyramid[2], support_mask))

        hypercorr_sqz4 = self.mean_keep_dims(hypercorr_sqz4, (-1, -2))

        target_size_3 = hypercorr_sqz3.shape[-4:-2]
        hypercorr_sqz4 = self.interpolate_query_dims(hypercorr_sqz4, target_size_3)
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43, _ = self.encoder_layer4to3((hypercorr_mix43, support_mask))

        target_size_2 = hypercorr_sqz2.shape[-4:-2]
        hypercorr_mix43 = self.interpolate_query_dims(hypercorr_mix43, target_size_2)
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432, _ = self.encoder_layer3to2((hypercorr_mix432, support_mask))

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.shape
        hypercorr_encoded = self.mean(self.reshape(hypercorr_mix432, (bsz, ch, ha, wa, -1)), 4)

        hypercorr_decoded = self.decoder1(hypercorr_encoded)

        upsample_size = (hypercorr_decoded.shape[-1] * 2, hypercorr_decoded.shape[-2] * 2) 
        hypercorr_decoded = self.interpolate(hypercorr_decoded, upsample_size)
        
        logit_mask = self.decoder2(hypercorr_decoded)

        return logit_mask
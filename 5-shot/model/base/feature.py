import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

avg_pool2d = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='SAME')

def extract_feat_vgg(img: Tensor, backbone: nn.Cell, feat_ids: list, bottleneck_ids: list = None, lids: list = None) -> list:
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(ops.copy(feat))
    return feats

def extract_feat_res(img: Tensor, backbone: nn.Cell, feat_ids: list, bottleneck_ids: list, lids: list) -> list:
    feats = []

    feat = backbone.conv1(img)
    feat = backbone.max_pool(feat)

    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat

        block = backbone.__getattr__('layer%d' % lid)[bid]

        feat = block.conv1(feat)
        feat = block.conv2(feat)
        feat = block.conv3(feat)

        if bid == 0:
            res = block.down_sample(res)

        feat = feat + res

        if hid + 1 in feat_ids:
            feats.append(feat.copy())

        feat = block.relu(feat)

    return feats


def as_extract_feat_res(img: Tensor, backbone: nn.Cell, feat_ids: list, bottleneck_ids: list, lids: list, pool: bool = False, pool_thr: int = 50) -> list:
    feats = []

    feat = backbone.conv1(img)
    feat = backbone.max_pool(feat)

    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        
        block = backbone.__getattr__('layer%d' % lid)[bid]

        feat = block.conv1(feat)
        feat = block.conv2(feat)
        feat = block.conv3(feat)

        if bid == 0:
            res = block.down_sample(res)

        feat = feat + res

        if hid + 1 in feat_ids:
            if pool and feat.shape[-1] >= pool_thr:
                feats.append(avg_pool2d(ops.copy(feat)))
            else:
                feats.append(ops.copy(feat))

        feat = block.relu(feat)

    return feats

def extract_feat_res_layer(img: Tensor, backbone: nn.Cell, feat_ids: list, bottleneck_ids: list, lids: list, layer_id_list: list) -> tuple:

    feats = []

    feat = backbone.conv1(img)
    feat = backbone.max_pool(feat)

    layer_i = 0
    layers = [feat]

    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        
        block = backbone.__getattr__('layer%d' % lid)[bid]

        feat = block.conv1(feat)
        feat = block.conv2(feat)
        feat = block.conv3(feat)

        if bid == 0:
            res = block.down_sample(res)

        feat = feat + res

        if hid + 1 in feat_ids:
            feats.append(ops.copy(feat))

        feat = block.relu(feat)

        if layer_i < len(layer_id_list) and hid + 1 == layer_id_list[layer_i]:
            layer_i += 1
            layers.append(ops.copy(feat))

    return feats, layers
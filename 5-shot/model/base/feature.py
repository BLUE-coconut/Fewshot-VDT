r""" Extracts intermediate features from given backbone network & layer ids """


def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
    r""" Extract intermediate features from VGG """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())
    return feats


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

    return feats



def extract_feat_res_layer(img, backbone, feat_ids, bottleneck_ids, lids, layer_id_list):
    r""" Extract intermediate features from ResNet"""

    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    layer_i = 0
    layers = [feat]

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        if hid + 1 == layer_id_list[layer_i] and layer_i < len(layer_id_list):
            layer_i += 1
            layers.append(feat.clone())

    return feats,layers
r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindvision.classification.models import resnet50, resnet101
# from mindvision.classification.models.vgg import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner
from .Sema_Trifuse import TriFusenet

class HypercorrSqueezeNetwork(nn.Cell):
    def __init__(self, backbone, use_original_imgsize):
        super(HypercorrSqueezeNetwork, self).__init__()

        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize

        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

        lids_tensor = ms.Tensor(self.lids, dtype=ms.int32)
        bincount = ops.bincount(lids_tensor)
        reversed_bincount = ops.flip(bincount, [0])
        cumsum = ops.cumsum(reversed_bincount, axis=0)
        self.stack_ids = cumsum[:3]
        
        self.backbone.set_train(False)
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.unsqueeze = ops.ExpandDims()
        self.interpolate = ops.ResizeBilinear
        self.argmax = ops.Argmax(axis=1)
        self.max = ops.Maximum()
        self.stack = ops.Stack(axis=0)

    def construct(self, query_img, support_img, support_mask):
        query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.mask_feature(support_feats, support_mask)
        corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)

        logit_mask = self.hpn_learner(corr)
        if not self.use_original_imgsize:
            logit_mask = ops.interpolate(logit_mask, 
                                       sizes=support_img.shape[2:], 
                                       mode='bilinear', 
                                       align_corners=True)

        return logit_mask

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = ops.interpolate(self.unsqueeze(support_mask.astype(ms.float32), 1), 
                                 sizes=feature.shape[2:], 
                                 mode='bilinear', 
                                 align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], 
                            batch['support_imgs'][:, s_idx], 
                            batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = (batch['org_query_imsize'][1].asnumpy()[0], 
                                batch['org_query_imsize'][0].asnumpy()[0])
                logit_mask = ops.interpolate(logit_mask, 
                                           sizes=org_qry_imsize, 
                                           mode='bilinear', 
                                           align_corners=True)

            logit_mask_agg += self.argmax(logit_mask)
            if nshot == 1: 
                return logit_mask_agg

        bsz = logit_mask_agg.shape[0]
        max_vote = logit_mask_agg.view(bsz, -1).max(axis=1)[0]
        max_vote = self.stack([max_vote, ops.ones_like(max_vote).astype(ms.int32)])
        max_vote = self.max(max_vote, axis=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.astype(ms.float32) / max_vote
        pred_mask = ops.where(pred_mask < 0.5, 
                            ops.zeros_like(pred_mask), 
                            ops.ones_like(pred_mask))

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.shape[0]
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).astype(ms.int32)

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.set_train(True)
        self.backbone.set_train(False)


class SEMA(nn.Cell):
    def __init__(self, backbone, use_original_imgsize):
        super(SEMA, self).__init__()

        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize

        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        
        # Convert to MindSpore Tensor operations
        lids_tensor = ms.Tensor(self.lids, dtype=ms.int32)
        bincount = ops.bincount(lids_tensor)
        reversed_bincount = ops.flip(bincount, [0])
        cumsum = ops.cumsum(reversed_bincount, axis=0)
        self.stack_ids = cumsum[:3]
        
        self.backbone.set_train(False)  # backbone frozen
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self.decoder = nn.SequentialCell([
            nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        ])
        self.fuse = TriFusenet()
        
        # Operations
        self.unsqueeze = ops.ExpandDims()
        self.interpolate = ops.ResizeBilinear
        self.argmax = ops.Argmax(axis=1)
        self.max = ops.Maximum()
        self.stack = ops.Stack(axis=0)

    def construct(self, query_img, query_img_th, query_img_d, support_img, support_img_th, support_img_d, support_mask):
        # Extract features from three modalities
        query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.mask_feature(support_feats, support_mask)

        query_feats_th = self.extract_feats(query_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats_th = self.extract_feats(support_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats_th = self.mask_feature(support_feats_th, support_mask)

        query_feats_d = self.extract_feats(query_img_d, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats_d = self.extract_feats(support_img_d, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats_d = self.mask_feature(support_feats_d, support_mask)

        corr = Correlation.multilayer_correlation_v1_0(query_feats, support_feats, self.stack_ids)
        corr_th = Correlation.multilayer_correlation_v1_0(query_feats_th, support_feats_th, self.stack_ids)
        corr_d = Correlation.multilayer_correlation_v1_0(query_feats_d, support_feats_d, self.stack_ids)

        logit_mask = self.hpn_learner(corr)
        logit_mask_d = self.hpn_learner(corr_d)
        logit_mask_th = self.hpn_learner(corr_th)
        
        logit_mask = self.fuse(logit_mask, logit_mask_th, logit_mask_d)
        logit_mask = self.decoder(logit_mask)

        if not self.use_original_imgsize:
            logit_mask = ops.interpolate(logit_mask, 
                                       sizes=support_img.shape[2:], 
                                       mode='bilinear', 
                                       align_corners=True)
        
        return logit_mask

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = ops.interpolate(self.unsqueeze(support_mask.astype(ms.float32), 1), 
                                 sizes=feature.shape[2:], 
                                 mode='bilinear', 
                                 align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], 
                            batch['support_imgs'][:, s_idx], 
                            batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = (batch['org_query_imsize'][1].asnumpy()[0], 
                                batch['org_query_imsize'][0].asnumpy()[0])
                logit_mask = ops.interpolate(logit_mask, 
                                           sizes=org_qry_imsize, 
                                           mode='bilinear', 
                                           align_corners=True)

            logit_mask_agg += self.argmax(logit_mask)
            if nshot == 1: 
                return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.shape[0]
        max_vote = logit_mask_agg.view(bsz, -1).max(axis=1)[0]
        max_vote = self.stack([max_vote, ops.ones_like(max_vote).astype(ms.int32)])
        max_vote = self.max(max_vote, axis=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.astype(ms.float32) / max_vote
        pred_mask = ops.where(pred_mask < 0.5, 
                            ops.zeros_like(pred_mask), 
                            ops.ones_like(pred_mask))

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.shape[0]
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).astype(ms.int32)

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.set_train(True)
        self.backbone.set_train(False)  # to prevent BN from learning data statistics with exponential averaging
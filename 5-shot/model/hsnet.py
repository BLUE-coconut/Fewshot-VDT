r""" Hypercorrelation Squeeze Network for Multi-modal 5-shot (MindSpore) """
from functools import reduce
from operator import add

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindvision.classification.models import resnet50, resnet101
# from mindvision.classification.models.vgg import vgg16 # 假设 vgg16 可用

# 假设以下模块已转换为 MindSpore
from .Sema_Trifuse import TriFusenet
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner


class SEMA_5shot(nn.Cell):
    def __init__(self, backbone, use_original_imgsize, shot):
        super(SEMA_5shot, self).__init__()

        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        self.shot = shot


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

        self.decoder = nn.SequentialCell([
            nn.Conv2d(64, 64, kernel_size=3, padding=1, pad_mode='pad', has_bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        ])
        self.fuse = TriFusenet()

        self.unsqueeze = ops.ExpandDims()
        self.interpolate_op = ops.ResizeBilinear(coordinate_transformation_mode="align_corners")
        self.argmax = ops.Argmax(axis=1)
        self.max = ops.Maximum()
        self.stack = ops.Stack(axis=0)
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.cast = ops.Cast()
        self.sum = ops.ReduceSum()
        self.div = ops.Div()


    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = self.interpolate_op(self.unsqueeze(self.cast(support_mask, ms.float32), 1),
                                       size=feature.shape[2:])
            features[idx] = features[idx] * mask
        return features

    def construct(self, query_img, query_img_th, query_img_d, support_imgs, support_img_ths, support_img_ds, support_masks):

        logit_mask_rgb_list = []
        logit_mask_th_list = []
        logit_mask_d_list = []

        for i in range(self.shot):

            support_img = support_imgs[:, i, :, :, :]
            support_img_th = support_img_ths[:, i, :, :, :]
            support_img_d = support_img_ds[:, i, :, :, :]
            support_mask = support_masks[:, i, :, :]

            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            query_feats_th = self.extract_feats(query_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            query_feats_d = self.extract_feats(query_img_d, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_th = self.extract_feats(support_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_d = self.extract_feats(support_img_d, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            support_feats = self.mask_feature(support_feats, support_mask)
            support_feats_th = self.mask_feature(support_feats_th, support_mask)
            support_feats_d = self.mask_feature(support_feats_d, support_mask)

            corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
            corr_th = Correlation.multilayer_correlation(query_feats_th, support_feats_th, self.stack_ids)
            corr_d = Correlation.multilayer_correlation(query_feats_d, support_feats_d, self.stack_ids)

            logit_mask_tmp = self.hpn_learner(corr)
            logit_mask_th_tmp = self.hpn_learner(corr_th)
            logit_mask_d_tmp = self.hpn_learner(corr_d)

            logit_mask_rgb_list.append(logit_mask_tmp)
            logit_mask_th_list.append(logit_mask_th_tmp)
            logit_mask_d_list.append(logit_mask_d_tmp)

        rgb_stacked = self.stack(logit_mask_rgb_list)
        th_stacked = self.stack(logit_mask_th_list)
        d_stacked = self.stack(logit_mask_d_list)

        rgb = self.div(self.sum(rgb_stacked, axis=0), self.cast(ms.Tensor(self.shot), ms.float32))
        th = self.div(self.sum(th_stacked, axis=0), self.cast(ms.Tensor(self.shot), ms.float32))
        d = self.div(self.sum(d_stacked, axis=0), self.cast(ms.Tensor(self.shot), ms.float32))

        logit_mask = self.fuse(rgb, th, d)
        logit_mask = self.decoder(logit_mask)

        if not self.use_original_imgsize:
            logit_mask = self.interpolate_op(logit_mask,
                                           size=support_img.shape[2:])

        return logit_mask

    def predict_mask_5shot(self, logit_mask, shot):

        if len(logit_mask.shape) == 5:
            logit_mask_agg = self.sum(self.argmax(logit_mask), axis=0)
        else:
            return self.argmax(logit_mask)

        bsz = logit_mask_agg.shape[0]
        flat_agg = logit_mask_agg.view(bsz, -1)
        max_vote = flat_agg.max(axis=1)[0]

        max_vote = self.stack([max_vote, self.ones_like(max_vote)])
        max_vote = self.max(max_vote, axis=0)[0].view(bsz, 1, 1)

        pred_mask = self.div(self.cast(logit_mask_agg, ms.float32), self.cast(max_vote, ms.float32))

        pred_mask = ops.where(pred_mask < 0.5,
                            self.zeros_like(pred_mask),
                            self.ones_like(pred_mask))

        return pred_mask

    def predict_mask_nshot(self, batch, nshot):

        if self.shot != nshot:
             raise ValueError("The model is SEMA_5shot, but predict_mask_nshot called with nshot={}".format(nshot))

        logit_mask = self.construct(
            batch['query_img'],
            batch['query_img_th'],
            batch['query_img_d'],
            batch['support_imgs'][:, :nshot],
            batch['support_img_ths'][:, :nshot],
            batch['support_img_ds'][:, :nshot],
            batch['support_masks'][:, :nshot])

        logit_mask_agg = self.argmax(logit_mask)

        if self.use_original_imgsize:
            org_qry_imsize_h = batch['org_query_imsize'][0].asnumpy()[0]
            org_qry_imsize_w = batch['org_query_imsize'][1].asnumpy()[0]
            org_qry_imsize = (org_qry_imsize_h, org_qry_imsize_w)

            pred_mask_interp = ops.interpolate(self.unsqueeze(self.cast(logit_mask_agg, ms.float32), 1),
                                           sizes=org_qry_imsize,
                                           mode='nearest')
            pred_mask = ops.squeeze(pred_mask_interp, axis=1)

        return pred_mask


    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.shape[0]
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).astype(ms.int32)

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.set_train(True)
        self.backbone.set_train(False)
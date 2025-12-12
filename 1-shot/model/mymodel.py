r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from .Trifuse import *
from .base.feature import extract_feat_vgg, extract_feat_res, extract_feat_res_layer
from .base.correlation import Correlation


class IFCNet(nn.Cell):
    def __init__(self, backbone, use_original_imgsize, dropout=0.0, vis=False):
        super(IFCNet, self).__init__()
        # add supervision
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        self.vis = vis
        

        if backbone == 'resnet50':
            from mindvision.classification.models import resnet50
            self.backbone = resnet50(pretrained=True)

            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 6, 3]
            self.multilevel_ch = [512, 1024, 2048]  # l4,l3,l2
        elif backbone == 'resnet101':
            from mindvision.classification.models import resnet101
            self.backbone = resnet101(pretrained=True)

            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 23, 3]
            self.multilevel_ch = [512, 1024, 2048]  # l2,l3,l4
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.nlevel = len(self.multilevel_ch)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nbottlenecks)])

        lids_array = np.array(self.lids)
        unique_values, counts = np.unique(lids_array, return_counts=True)

        reversed_counts = counts[::-1]
        cumsum_counts = np.cumsum(reversed_counts)
        self.stack_ids = cumsum_counts[:3]
        
        self.layer_ids = np.array([0] + self.nbottlenecks[1:]).cumsum()
        self.nbottlenecks = self.nbottlenecks[1:]
        self.backbone.set_train(False)
        self.multi_smi_fuse = nn.CellList([ConvModule(self.nbottlenecks[i] * 3, self.nbottlenecks[i], 1)
                                           for i in range(len(self.nbottlenecks))])  # correspond to l2,l3,l4

        self.tri_mid_ch = 128
        self.tri_out_ch = 64
        self.TriFusel4 = TriCorr_v2_6(in_ch=self.multilevel_ch[-1], mid_ch=self.tri_mid_ch, out_ch=self.tri_out_ch,
                                      dropout=dropout)
        self.TriFusel3 = TriCorr_v2_6(in_ch=self.multilevel_ch[-2], mid_ch=self.tri_mid_ch, out_ch=self.tri_out_ch,
                                      dropout=dropout)

        self.Fusion = ConvModule(self.tri_out_ch + self.nbottlenecks[-1], self.tri_mid_ch, 1)
        self.deblock1 = ConvModule(self.tri_out_ch + self.tri_mid_ch + self.nbottlenecks[-2], self.tri_out_ch, 1)
        self.deblock2 = ConvModule(self.tri_out_ch + self.nbottlenecks[-3], self.tri_out_ch, 1)

        self.cross_entropy_loss = nn.transformer.CrossEntropyLoss()

        self.side3 = ConvModule(self.tri_mid_ch, 1, 3, mode="relu")
        self.side2 = ConvModule(self.tri_out_ch, 1, 3, mode="relu")
        self.side1 = ConvModule(self.tri_out_ch, 1, 3, mode="relu")

        self.unsqueeze = ops.ExpandDims()
        self.cat = ops.Concat(axis=1)
        self.interpolate = ops.ResizeBilinear
        self.argmax = ops.Argmax(axis=1)
        self.max = ops.Maximum()
        self.stack = ops.Stack(axis=0)
        self.stop_gradient = ops.stop_gradient
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()

    def _upsample_like(self, src, tar):
        tar_h, tar_w = tar.shape[2], tar.shape[3]
        resize_op = P.ResizeBilinear(size=(tar_h, tar_w))
        return resize_op(src)


    def construct(self, query_img, query_img_th, query_img_d, support_img, support_img_th, support_img_d, support_mask):

        query_feats = self.extract_feats(query_img, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.extract_feats(support_img, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats = self.mask_feature(support_feats, support_mask)

        query_feats_th = self.extract_feats(query_img_th, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats_th = self.extract_feats(support_img_th, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats_th = self.mask_feature(support_feats_th, support_mask)

        query_feats_d = self.extract_feats(query_img_d, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats_d = self.extract_feats(support_img_d, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        support_feats_d = self.mask_feature(support_feats_d, support_mask)

        query_feats = [query_feats[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
        support_feats = [support_feats[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
        query_feats_th = [query_feats_th[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
        support_feats_th = [support_feats_th[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
        query_feats_d = [query_feats_d[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
        support_feats_d = [support_feats_d[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]

        multi_sim = []
        if self.vis:
            vsim_maps = []
            dsim_maps = []
            tsim_maps = []
            
        for idx, (qv_layer, sv_layer, qd_layer, sd_layer, qt_layer, st_layer) in enumerate(
                zip(query_feats, support_feats, query_feats_d,
                    support_feats_d, query_feats_th, support_feats_th)):
            vsim = []
            dsim = []
            tsim = []
            for iidx, (qv, sv, qd, sd, qt, st) in enumerate(
                    zip(qv_layer, sv_layer, qd_layer, sd_layer, qt_layer, st_layer)):
                vsim.append(Correlation.multi_similarity(qv, sv))
                dsim.append(Correlation.multi_similarity(qd, sd))
                tsim.append(Correlation.multi_similarity(qt, st))

            trisim = self.multi_smi_fuse[idx](self.cat(vsim + dsim + tsim))
            multi_sim.append(trisim)
            if self.vis:
                vsim_maps.append(vsim)
                dsim_maps.append(dsim)
                tsim_maps.append(tsim)

        Trimap_l4 = self.TriFusel4([query_feats_d[-1][-1], query_feats[-1][-1], query_feats_th[-1][-1]],
                                   [support_feats_d[-1][-1], support_feats[-1][-1], support_feats_th[-1][-1]])  # features = [fV,fD,fT]
        Trimap_l4 = self.cat(Trimap_l4)
        if self.vis:
            vis_l4 = [query_feats[-1][-1], query_feats_d[-1][-1], query_feats_th[-1][-1], Trimap_l4]

        Trimap_l3 = self.TriFusel3([query_feats_d[-2][-1], query_feats[-2][-1], query_feats_th[-2][-1]],
                                   [support_feats_d[-2][-1], support_feats[-2][-1], support_feats_th[-2][-1]])
        Trimap_l3 = self.cat(Trimap_l3)
        if self.vis:
            vis_l3 = [query_feats[-2][-1], query_feats_d[-2][-1], query_feats_th[-2][-1], Trimap_l3]

        logit_masks = []

        out3 = self.Fusion(self.cat([Trimap_l4, multi_sim[-1]]))
        logit_masks.append(self.side3(out3))

        out2 = self.deblock1(self.cat([Trimap_l3, self._upsample_like(out3, multi_sim[-2]), multi_sim[-2]]))
        logit_masks.append(self.side2(out2))

        out1 = self.deblock2(self.cat([self._upsample_like(out2, multi_sim[-3]), multi_sim[-3]]))
        logit_masks.append(self.side1(out1))


        if not self.use_original_imgsize:
            for idx, logit_mask in enumerate(logit_masks):
                tar_h, tar_w = support_img.shape[2], support_img.shape[3]
                resize = P.ResizeBilinear(size=(tar_h, tar_w))
                logit_masks[idx] = resize(logit_mask)
        if self.vis:
            return logit_masks, vsim_maps, dsim_maps, tsim_maps, multi_sim, vis_l3, vis_l4
        else:
            return logit_masks

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            resize = ops.ResizeBilinear(feature.shape[2:], align_corners=True)
            mask = resize(support_mask.astype(ms.float32))
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

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
                              self.zeros(pred_mask.shape, ms.float32),
                              self.ones(pred_mask.shape, ms.float32))

        return pred_mask

    def compute_objective(self, logit_masks, gt_mask):
        if isinstance(logit_masks, list):
            loss = 0
            for logit_mask in logit_masks:
                bsz = logit_mask.shape[0]
                logit_mask = logit_mask.view(-1)
                gt_mask = gt_mask.view(-1).astype(ms.int32)
                logit = [gt_mask.shape, 2]
                loss = loss + self.cross_entropy_loss(logits=logit, input_mask=logit_mask, label=gt_mask)
        else:
            logit_mask = logit_masks
            bsz = logit_mask.shape[0]
            logit_mask = logit_mask.view(-1)
            gt_mask = gt_mask.view(-1).astype(ms.int32)
            logit = ms.Tensor(2,gt_mask.shape[0]).astype(mstype.float32)
            loss = self.cross_entropy_loss(logits=logit, input_mask=logit_mask, label=gt_mask)

        return loss

    def train_mode(self):
        self.set_train(True)
        self.backbone.set_train(False)
from functools import reduce
from operator import add
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

# 假定 Trifuse 中定义了 ConvModule, TriCorr_v2_6 等（与 model.py 相同）
from .Trifuse import *
from .base.feature import extract_feat_vgg, extract_feat_res, extract_feat_res_layer
from .base.correlation import Correlation


class IFCNet_5shot(nn.Cell):
    def __init__(self, backbone, use_original_imgsize, dropout=0.2, vis=False, shot=1):
        super(IFCNet_5shot, self).__init__()
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        self.vis = vis
        self.shot = shot

        if backbone == 'resnet50':
            from mindvision.classification.models import resnet50
            self.backbone = resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 6, 3]
            self.multilevel_ch = [512, 1024, 2048]
        elif backbone == 'resnet101':
            from mindvision.classification.models import resnet101
            self.backbone = resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 23, 3]
            self.multilevel_ch = [512, 1024, 2048]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.nlevel = len(self.multilevel_ch)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nbottlenecks)])
        self.layer_ids = np.array([0] + self.nbottlenecks[1:]).cumsum()
        self.nbottlenecks = self.nbottlenecks[1:]

        try:
            self.backbone.set_train(False)
        except Exception:
            pass

        self.multi_smi_fuse = nn.CellList([
            ConvModule(self.nbottlenecks[i] * 3, self.nbottlenecks[i], 1)
            for i in range(len(self.nbottlenecks))
        ])

        self.tri_mid_ch = 128
        self.tri_out_ch = 64
        self.TriFusel4 = TriCorr_v2_6(in_ch=self.multilevel_ch[-1], mid_ch=self.tri_mid_ch,
                                      out_ch=self.tri_out_ch, dropout=dropout)
        self.TriFusel3 = TriCorr_v2_6(in_ch=self.multilevel_ch[-2], mid_ch=self.tri_mid_ch,
                                      out_ch=self.tri_out_ch, dropout=dropout)

        self.Fusion = ConvModule(self.tri_out_ch + self.nbottlenecks[-1], self.tri_mid_ch, 1)
        self.deblock1 = ConvModule(self.tri_out_ch + self.tri_mid_ch + self.nbottlenecks[-2], self.tri_out_ch, 1)
        self.deblock2 = ConvModule(self.tri_out_ch + self.nbottlenecks[-3], self.tri_out_ch, 1)


        self.cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

        self.side3 = ConvModule(self.tri_mid_ch, 1, 3, mode="relu")
        self.side2 = ConvModule(self.tri_out_ch, 1, 3, mode="relu")
        self.side1 = ConvModule(self.tri_out_ch, 1, 3, mode="relu")

        self.cat = ops.Concat(axis=1)
        self.stack = ops.Stack(axis=0)
        self.stop_gradient = ops.stop_gradient
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()
        self.argmax = ops.Argmax(axis=1)
        self.max = ops.Maximum()
        self.unsqueeze = ops.ExpandDims()
        self.resize_op_class = P.ResizeBilinear
        self.reduce_mean = P.ReduceMean(keep_dims=False)

    def _upsample_like(self, src, tar):
        tar_h, tar_w = tar.shape[2], tar.shape[3]
        resize_op = P.ResizeBilinear(size=(tar_h, tar_w))
        return resize_op(src)

    def construct(self, query_img, query_img_th, query_img_d,
                  support_imgs, support_img_ths, support_img_ds, support_masks):
        """
        Args (expected shapes):
          - query_img: (B, C, H, W)
          - support_imgs: (B, S, C, H, W)
          - support_masks: (B, S, H, W)
          - support_img_ths, support_img_ds: same shape as support_imgs
        Returns:
          - logit_masks: list of 3 side outputs [(B,2,H,W), ...]
        """

        query_feats = self.extract_feats(query_img, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        query_feats_th = self.extract_feats(query_img_th, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        query_feats_d = self.extract_feats(query_img_d, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

        query_feats = [query_feats[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
        query_feats_th = [query_feats_th[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
        query_feats_d = [query_feats_d[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]

        per_shot_multi_sims = []
        per_shot_Trimap_l3 = []
        per_shot_Trimap_l4 = []

        B = query_img.shape[0]
        S = support_imgs.shape[1]

        for s_idx in range(S):

            support_img = support_imgs[:, s_idx, :, :, :]
            support_img_th = support_img_ths[:, s_idx, :, :, :]
            support_img_d = support_img_ds[:, s_idx, :, :, :]
            support_mask = support_masks[:, s_idx, :, :]

            support_feats = self.extract_feats(support_img, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_th = self.extract_feats(support_img_th, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_d = self.extract_feats(support_img_d, self.backbone.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            support_feats = self.mask_feature(support_feats, support_mask)
            support_feats_th = self.mask_feature(support_feats_th, support_mask)
            support_feats_d = self.mask_feature(support_feats_d, support_mask)

            support_feats = [support_feats[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
            support_feats_th = [support_feats_th[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]
            support_feats_d = [support_feats_d[self.layer_ids[i]:self.layer_ids[i + 1]] for i in range(0, len(self.layer_ids) - 1)]

            shot_multi_sim = []
            for idx, (qv_layer, sv_layer, qd_layer, sd_layer, qt_layer, st_layer) in enumerate(
                    zip(query_feats, support_feats, query_feats_d, support_feats_d, query_feats_th, support_feats_th)):
                vsim = []
                dsim = []
                tsim = []
                for (qv, sv, qd, sd, qt, st) in zip(qv_layer, sv_layer, qd_layer, sd_layer, qt_layer, st_layer):
                    vsim.append(Correlation.multi_similarity(qv, sv))
                    dsim.append(Correlation.multi_similarity(qd, sd))
                    tsim.append(Correlation.multi_similarity(qt, st))

                trisim = self.multi_smi_fuse[idx](self.cat(vsim + dsim + tsim))

                shot_multi_sim.append(self.unsqueeze(trisim, 1))
            per_shot_multi_sims.append(shot_multi_sim)

            Trimap_l4 = self.TriFusel4([query_feats_d[-1][-1], query_feats[-1][-1], query_feats_th[-1][-1]],
                                       [support_feats_d[-1][-1], support_feats[-1][-1], support_feats_th[-1][-1]])
            Trimap_l4 = self.cat(Trimap_l4)
            per_shot_Trimap_l4.append(self.unsqueeze(Trimap_l4, 1))

            Trimap_l3 = self.TriFusel3([query_feats_d[-2][-1], query_feats[-2][-1], query_feats_th[-2][-1]],
                                       [support_feats_d[-2][-1], support_feats[-2][-1], support_feats_th[-2][-1]])
            Trimap_l3 = self.cat(Trimap_l3)
            per_shot_Trimap_l3.append(self.unsqueeze(Trimap_l3, 1))

        num_levels = len(per_shot_multi_sims[0])
        multi_sims = []
        for lvl in range(num_levels):

            concat_list = [per_shot_multi_sims[s][lvl] for s in range(S)]
            concat_shots = ops.Concat(axis=1)(concat_list)

            mean_over_shot = self.reduce_mean(concat_shots, 1)

            multi_sims.append(mean_over_shot)

        Trimap_l4s = self.reduce_mean(ops.Concat(axis=1)(per_shot_Trimap_l4), 1)  # -> (B, C, H, W)
        Trimap_l3s = self.reduce_mean(ops.Concat(axis=1)(per_shot_Trimap_l3), 1)  # -> (B, C, H, W)

        logit_masks = []

        out3 = self.Fusion(self.cat([Trimap_l4s, multi_sims[-1]]))
        logit_masks.append(self.side3(out3))

        out2 = self.deblock1(self.cat([Trimap_l3s, self._upsample_like(out3, multi_sims[-2]), multi_sims[-2]]))
        logit_masks.append(self.side2(out2))

        out1 = self.deblock2(self.cat([self._upsample_like(out2, multi_sims[-3]), multi_sims[-3]]))
        logit_masks.append(self.side1(out1))

        if not self.use_original_imgsize:
            tar_h, tar_w = support_imgs.shape[3], support_imgs.shape[4]
            for i, lm in enumerate(logit_masks):
                resize = P.ResizeBilinear(size=(tar_h, tar_w))
                logit_masks[i] = resize(lm)

        return logit_masks

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            h, w = feature.shape[2], feature.shape[3]
            resize = P.ResizeBilinear(size=(h, w))
            if support_mask.ndim == 3:
                mask = support_mask.expand_dims(1).astype(ms.float32)
            else:
                mask = support_mask.astype(ms.float32)
            mask_resized = resize(mask)
            features[idx] = features[idx] * mask_resized
        return features

    def predict_mask_nshot(self, batch, nshot):

        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['query_img_th'], batch['query_img_d'],
                              batch['support_imgs'][:, s_idx], batch['support_img_ths'][:, s_idx],
                              batch['support_img_ds'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = (batch['org_query_imsize'][1].asnumpy()[0], batch['org_query_imsize'][0].asnumpy()[0])
                resize = P.ResizeBilinear(size=(org_qry_imsize[0], org_qry_imsize[1]))
                logit_mask = [resize(lm) for lm in logit_mask]

            logit_mask_agg += self.argmax(logit_mask[0])
            if nshot == 1:
                return logit_mask_agg

        bsz = logit_mask_agg.shape[0]
        max_vote = logit_mask_agg.view(bsz, -1).max(axis=1)[0]
        max_vote = self.stack([max_vote, ops.Ones()(max_vote.shape, max_vote.dtype).astype(ms.int32)])
        max_vote = self.max(max_vote, axis=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.astype(ms.float32) / max_vote
        pred_mask = ops.where(pred_mask < 0.5, self.zeros(pred_mask.shape, ms.float32), self.ones(pred_mask.shape, ms.float32))
        return pred_mask

    def compute_objective(self, logit_masks, gt_mask):
        """
        logit_masks: list of (B,2,H,W) or a single tensor
        gt_mask: (B,H,W)
        """
        if isinstance(logit_masks, list):
            loss = 0
            for lm in logit_masks:
                bsz = lm.shape[0]
                logit = lm.view(bsz, 2, -1)
                label = gt_mask.view(bsz, -1).astype(ms.int32)
                loss = loss + self.cross_entropy_loss(logit.transpose(0, 2, 1).reshape(-1, 2), label.reshape(-1))
            return loss
        else:
            lm = logit_masks
            bsz = lm.shape[0]
            logit = lm.view(bsz, 2, -1)
            label = gt_mask.view(bsz, -1).astype(ms.int32)
            loss = self.cross_entropy_loss(logit.transpose(0, 2, 1).reshape(-1, 2), label.reshape(-1))
            return loss

    def train_mode(self):
        self.set_train(True)
        try:
            self.backbone.set_train(False)
        except Exception:
            pass

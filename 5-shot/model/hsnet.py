r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .Sema_Trifuse import TriFusenet
from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner


class SEMA_5shot(nn.Module):
    def __init__(self, backbone, use_original_imgsize, shot):
        super(SEMA_5shot, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        self.shot = shot
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.decoder = nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 2, (3, 3), padding=(1, 1), bias=True))
        self.fuse = TriFusenet()

    def forward(self, query_img, query_img_th, query_img_d,  support_imgs, support_img_ths, support_img_ds,  support_masks):

        logit_mask_rgb_list = []
        logit_mask_th_list = []
        logit_mask_d_list = []
        for i in range(self.shot):
            support_img = support_imgs[:, i, :, :, :]
            support_img_th = support_img_ths[:, i, :, :, :]
            support_img_d = support_img_ds[:, i, :, :, :]
            support_mask = support_masks[:, i, :, :]

            with torch.no_grad():
                query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                query_feats_th = self.extract_feats(query_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                query_feats_d = self.extract_feats(query_img_d, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

                support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats_th = self.extract_feats(support_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
                support_feats_d = self.extract_feats(support_img_d, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

                support_feats = self.mask_feature(support_feats, support_mask.clone())
                support_feats_th = self.mask_feature(support_feats_th, support_mask.clone())
                support_feats_d = self.mask_feature(support_feats_d, support_mask.clone())

                corr = Correlation.multilayer_correlation(query_feats, support_feats, self.stack_ids)
                corr_th = Correlation.multilayer_correlation(query_feats_th, support_feats_th, self.stack_ids)
                corr_d = Correlation.multilayer_correlation(query_feats_d, support_feats_d, self.stack_ids)

            logit_mask_tmp = self.hpn_learner(corr)
            logit_mask_th_tmp = self.hpn_learner(corr_th)
            logit_mask_d_tmp = self.hpn_learner(corr_d)

            logit_mask_rgb_list.append(logit_mask_tmp)
            logit_mask_th_list.append(logit_mask_th_tmp)
            logit_mask_d_list.append(logit_mask_d_tmp)

        if self.shot > 1:
            rgb = logit_mask_rgb_list[0]
            th = logit_mask_th_list[0]
            d = logit_mask_d_list[0]
            for i in range(1, len(logit_mask_rgb_list)):
                rgb += logit_mask_rgb_list[i]
            rgb /= len(logit_mask_rgb_list)
            for j in range(1, len(logit_mask_th_list)):
                th += logit_mask_th_list[j]
            th /= len(logit_mask_th_list)
            for k in range(1, len(logit_mask_d_list)):
                d += logit_mask_d_list[k]
            d /= len(logit_mask_d_list)

        logit_mask = self.fuse(rgb, th, d)
        logit_mask = self.decoder(logit_mask)

        if not self.use_original_imgsize:
            logit_mask = F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True)

        return logit_mask

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_5shot(self, logit_mask, shot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for i in range(shot):
            logit_mask_agg += logit_mask[i].argmax(dim=1).clone()

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

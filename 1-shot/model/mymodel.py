r""" Hypercorrelation Squeeze Network """
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

#from .Trifusemodel import TriFusenet
from .Trifuse import *
from .base.feature import extract_feat_vgg, extract_feat_res, extract_feat_res_layer
from .base.correlation import Correlation,Correlation_V0

import numpy as np


class HS_VDT_v2_10(nn.Module):
    def __init__(self, backbone, use_original_imgsize, dropout=0.0, vis = False):
        super().__init__()
        # add supervision
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            self.nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.multilevel_ch = [256,512,512] # l4,l3,l2
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 6, 3]
            self.multilevel_ch = [512,1024,2048] # l4,l3,l2
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            self.nbottlenecks = [3, 4, 23, 3]
            self.multilevel_ch = [512,1024,2048] # l2,l3,l4
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.nlevel = len(self.multilevel_ch)
        self.vis = vis
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), self.nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nbottlenecks)])
        # self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.layer_ids = np.array([0]+self.nbottlenecks[1:]).cumsum()
        self.nbottlenecks = self.nbottlenecks[1:]
        self.backbone.eval() # backbone frozen

        # self.corr_mid = corr_mid*3 if len(corr_mid)==1 else corr_mid # correspond to l2,l3,l4

        self.multi_smi_fuse = nn.ModuleList([ConvModule(self.nbottlenecks[i]*3,self.nbottlenecks[i],1) 
                                             for i in range(len(self.nbottlenecks))]) # correspond to l2,l3,l4
        

        self.tri_mid_ch=128
        self.tri_out_ch=64
        self.TriFusel4 = TriCorr_v2_6(in_ch=self.multilevel_ch[-1],mid_ch=self.tri_mid_ch,out_ch=self.tri_out_ch,dropout=dropout)
        self.TriFusel3 = TriCorr_v2_6(in_ch=self.multilevel_ch[-2],mid_ch=self.tri_mid_ch,out_ch=self.tri_out_ch,dropout=dropout)
        #self.TriFusel2 = TriCorr_v2_3(in_ch=self.multilevel_ch[-1],mid_ch=self.tri_mid_ch,out_ch=self.tri_out_ch,dropout=0.2)

        self.Fusion = ConvModule(self.tri_out_ch+self.nbottlenecks[-1],self.tri_mid_ch,1)
        self.deblock1 = ConvModule(self.tri_out_ch+self.tri_mid_ch+self.nbottlenecks[-2],self.tri_out_ch,1)
        self.deblock2 = ConvModule(self.tri_out_ch+self.nbottlenecks[-3],self.tri_out_ch,1)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self.side3 = ConvModule(self.tri_mid_ch,2,3,mode="relu")
        self.side2 = ConvModule(self.tri_out_ch,2,3,mode="relu")
        self.side1 = ConvModule(self.tri_out_ch,2,3,mode="relu")
        #self.side0 = ConvModule(2*3,2,1,mode="sigmoid")

        

    def _upsample_like(self, src, tar):
        src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

        return src

    def forward(self, query_img, query_img_th, query_img_d,  support_img, support_img_th, support_img_d, support_mask):
        with torch.no_grad():
            # only the output of last layer in each level is used
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.mask_feature(support_feats, support_mask.clone())

            query_feats_th = self.extract_feats(query_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_th = self.extract_feats(support_img_th, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_th = self.mask_feature(support_feats_th, support_mask.clone())

            query_feats_d = self.extract_feats(query_img_d, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_d = self.extract_feats(support_img_d, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats_d = self.mask_feature(support_feats_d, support_mask.clone())

        ### Here all support feats are masked
        
        

         # pick l2, l3, l4
        query_feats = [query_feats[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        support_feats = [support_feats[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        query_feats_th = [query_feats_th[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        support_feats_th = [support_feats_th[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        query_feats_d = [query_feats_d[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        support_feats_d = [support_feats_d[self.layer_ids[i]:self.layer_ids[i+1]] for i in range(0,len(self.layer_ids)-1)]
        
        multi_sim = []
        if self.vis:
            vsim_maps = []
            dsim_maps = []
            tsim_maps = []
        for idx,(qv_layer,sv_layer,qd_layer,sd_layer,qt_layer,st_layer) in enumerate(zip(query_feats,support_feats,query_feats_d,
                                                 support_feats_d,query_feats_th,support_feats_th)):
            # 3 levels in total
            vsim = []
            dsim = []
            tsim = []
            for iidx, (qv,sv,qd,sd,qt,st) in enumerate(zip(qv_layer,sv_layer,qd_layer,sd_layer,qt_layer,st_layer)):             
                vsim.append(Correlation.multi_similarity(qv,sv))
                dsim.append(Correlation.multi_similarity(qd,sd))
                tsim.append(Correlation.multi_similarity(qt,st))
                
            trisim = self.multi_smi_fuse[idx](torch.cat(vsim+dsim+tsim,dim=1))
            multi_sim.append(trisim)
            if self.vis:
                vsim_maps.append(vsim)
                dsim_maps.append(dsim)
                tsim_maps.append(tsim)
            
        
        # multi_sim ->[l2,l3,l4] each level is (bsz,1,qH,qW)
             
        
        Trimap_l4 = self.TriFusel4([query_feats[-1][-1],query_feats_d[-1][-1],query_feats_th[-1][-1]],
                                 [support_feats[-1][-1],support_feats_d[-1][-1],support_feats_th[-1][-1]]) # features = [fV,fD,fT]
        Trimap_l4 = torch.cat(Trimap_l4,dim=1)
        if(self.vis):vis_l4 = [query_feats[-1][-1],query_feats_d[-1][-1],query_feats_th[-1][-1],Trimap_l4]


        Trimap_l3 = self.TriFusel3([query_feats[-2][-1],query_feats_d[-2][-1],query_feats_th[-2][-1]],
                                 [support_feats[-2][-1],support_feats_d[-2][-1],support_feats_th[-2][-1]])
        Trimap_l3 = torch.cat(Trimap_l3,dim=1)
        if(self.vis):vis_l3 = [query_feats[-2][-1],query_feats_d[-2][-1],query_feats_th[-2][-1],Trimap_l3]

        logit_masks=[]

        out3 = self.Fusion(torch.cat([Trimap_l4,multi_sim[-1]],dim=1))
        logit_masks.append(self.side3(out3))

        out2 = self.deblock1(torch.cat([Trimap_l3,self._upsample_like(out3,multi_sim[-2]),multi_sim[-2]],dim=1))
        logit_masks.append(self.side2(out2))

        out1 = self.deblock2(torch.cat([self._upsample_like(out2,multi_sim[-3]),multi_sim[-3]],dim=1))
        logit_masks.append(self.side1(out1))

        #logit_masks.append(self.side0(torch.cat([out1,out2,out3],dim=1)))
        

        if not self.use_original_imgsize:
            logit_masks=[F.interpolate(logit_mask, support_img.size()[2:], mode='bilinear', align_corners=True) for logit_mask in logit_masks]
                
        # decoder output 2 channel: indicate probabilities of foreground and background
        if self.vis:
            return logit_masks,vsim_maps,dsim_maps,tsim_maps,multi_sim,vis_l3,vis_l4
        else:
            return logit_masks

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            # inferencing should do nshot times for each support image
            logit_mask = self(batch['query_img'], batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx])

            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            # vote for background or target:
            
            if nshot == 1: return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        # max_vote>=1
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        # channel-0 present background and channel-1 present target 
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_masks, gt_mask):
        
        if isinstance(logit_masks,list):
            loss = 0 
            for logit_mask in logit_masks:
                bsz = logit_mask.size(0)
                logit_mask = logit_mask.view(bsz, 2, -1)  
                gt_mask = gt_mask.view(bsz, -1).long()
                loss = loss + self.cross_entropy_loss(logit_mask, gt_mask)

        else:      
            logit_mask = logit_masks      
            bsz = logit_mask.size(0)
            logit_mask = logit_mask.view(bsz, 2, -1)  
            gt_mask = gt_mask.view(bsz, -1).long()
            loss = self.cross_entropy_loss(logit_mask, gt_mask)

        return loss

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging


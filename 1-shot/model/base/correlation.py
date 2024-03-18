r""" Provides functions thqt builds/manipulates correlation tensors """
import torch
import torch.nn.functional as F

def cosine_similarity(query_f,support_f,support_mask = None):
    eps = 1e-5
    if not support_mask is None:
        mask = F.interpolate(support_mask.unsqueeze(1).float(), support_f.size()[2:], mode='bilinear', align_corners=True)
        support_f = support_f * mask

    bsz, ch, hs, ws = support_f.size()
    support_f = support_f.view(bsz, ch, -1) # bsz, ch, hw
    support_f = support_f / (support_f.norm(dim=1, p=2, keepdim=True) + eps)

    bsz, ch, hq, wq = query_f.size()
    query_f = query_f.view(bsz, ch, -1)# bsz, ch, hw
    query_f = query_f / (query_f.norm(dim=1, p=2, keepdim=True) + eps)

    corr = torch.bmm(support_f.transpose(1, 2), query_f)
    corr = corr.clamp(min=0)
    corr = corr.mean(dim=1,keepdim=True).view(bsz, -1, hq, wq)# bsz, 1, h, w
    
    return corr # bsz, 1, h, w


class Correlation:
    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]

    
    @classmethod
    def multi_similarity(cls, query_feats, support_feats, support_mask = None):
        '''if isinstance(query_feats,list):
            query_feats = torch.cat(query_feats,dim=1)
        if isinstance(support_feats,list):
            support_feats = torch.cat(support_feats,dim=1)'''

        cos_sim = cosine_similarity(query_f=query_feats,support_f=support_feats,support_mask=support_mask)

        return cos_sim





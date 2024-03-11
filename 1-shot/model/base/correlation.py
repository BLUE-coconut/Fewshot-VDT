r""" Provides functions thqt builds/manipulates correlation tensors """
import torch
import torch.nn.functional as F

def is_binary_matrix(matrix):
    pd = (matrix<1) & (matrix>0)
    return pd.sum()==0

def pro_binary_mask(matrix):
    if(not is_binary_matrix(matrix)):
        matrix[matrix>=0.5] = 1
        matrix[matrix<0.5] = 0
    return matrix.float()

def MAP(feature, mask):
    exp = 1e-5
    mask = pro_binary_mask(mask)
    mask = F.interpolate(mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
    pix_cnt = mask.sum(dim = (2,3))
    pix_avg = (feature*mask).sum(dim = (2,3))

    return pix_avg/(pix_cnt+exp)

def proto_corr(query_f,support_f,support_mask):
    eps = 1e-5
    # query_f:(bs,c,hq,wq)
    # support_f:(bs,c,hs,ws)
    support_proto = MAP(support_f, support_mask) # support_proto:(bsz,c)
    support_proto = support_proto.unsqueeze(2) # support_proto:(bsz,c,1)
    #bsz, ch, _= support_f.size()
    support_proto = support_proto / (support_proto.norm(dim=1, p=2, keepdim=True) + eps)

    bsz, ch, hq, wq = query_f.size()
    query_f = query_f.view(bsz, ch, -1) # query_f:(bsz,c,h*w)
    query_f = query_f / (query_f.norm(dim=1, p=2, keepdim=True) + eps)

    corr = torch.bmm(query_f.transpose(1, 2), support_proto).view(bsz, hq, wq)
    corr = corr.clamp(min=0)

    return corr # (bsz, hq, wq)

def priormap_generation(query_f,support_f,support_mask = None):
    eps = 1e-5
    if not support_mask is None:
        mask = F.interpolate(support_mask.unsqueeze(1).float(), support_f.size()[2:], mode='bilinear', align_corners=True)
        support_f = support_f * mask

    bsz, ch, hs, ws = support_f.size()
    support_f = support_f.view(bsz, ch, -1)
    support_f = support_f / (support_f.norm(dim=1, p=2, keepdim=True) + eps)

    bsz, ch, hq, wq = query_f.size()
    query_f = query_f.view(bsz, ch, -1)
    query_f = query_f / (query_f.norm(dim=1, p=2, keepdim=True) + eps)

    corr = torch.bmm(query_f.transpose(1, 2), support_f).view(bsz, -1, hq, wq)
    corr = torch.max(corr,dim=1,keepdim=True)
    corr = corr.clamp(min=0)

    return corr # (bsz, 1, hq, wq)

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

class Correlation_V0:

    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hs, ws = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, hq, wq = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, hq, wq, hs, ws)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]


class Correlation:
    @classmethod
    def multilayer_correlation_v1_0(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5

        corrs = []
                    

        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hs, ws = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, hq, wq = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, hq, wq, hs, ws)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]
    

    @classmethod
    def multilayer_correlation_v1_1(cls, query_feats, corr_feats, stack_ids):
        eps = 1e-5

        corrs = []

        for idx, support_feat in enumerate(query_feats):
            if(idx<stack_ids[2]-stack_ids[1]):
                corr_feat = corr_feats[2]
            
            elif(idx<stack_ids[2]-stack_ids[0]):
                corr_feat = corr_feats[1]

            elif(idx<stack_ids[2]):
                corr_feat = corr_feats[0]
            
            bsz, ch, hs, ws = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, hq, wq = corr_feat.size()
            corr_feat = corr_feat.view(bsz, ch, -1)
            corr_feat = corr_feat / (corr_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(corr_feat.transpose(1, 2), support_feat).view(bsz, hq, wq, hs, ws)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]

    @classmethod
    def multilayer_proto_corr(cls, query_feats, support_feats, support_mask, stack_ids):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            corrs.append(proto_corr(query_feat,support_feat,support_mask)) # (bsz, hq, wq)

        # corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        # corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        # corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()
        # (bsz, Cn, hq, wq)

        return corrs  #[corr_l4, corr_l3, corr_l2]
    
    @classmethod
    def multilayer_priormap_corr(cls, query_feats, support_feats, stack_ids, support_mask = None):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            corrs.append(priormap_generation(query_feat,support_feat,support_mask)) # (bsz, hq, wq)

        return corrs  #[corr_l4, corr_l3, corr_l2]
    
    @classmethod
    def multi_similarity(cls, query_feats, support_feats, support_mask = None):
        '''if isinstance(query_feats,list):
            query_feats = torch.cat(query_feats,dim=1)
        if isinstance(support_feats,list):
            support_feats = torch.cat(support_feats,dim=1)'''

        cos_sim = cosine_similarity(query_f=query_feats,support_f=support_feats,support_mask=support_mask)

        return cos_sim





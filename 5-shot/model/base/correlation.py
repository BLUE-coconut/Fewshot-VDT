import mindspore
import mindspore.ops as ops
from mindspore import Tensor


def cosine_similarity(query_f, support_f, support_mask=None):
    eps = 1e-5
    
    if support_mask is not None:
        mask_float = ops.expand_dims(support_mask, 1).astype(mindspore.float32)
        mask = ops.interpolate(mask_float, sizes=support_f.shape[2:], mode='bilinear', coordinate_transformation_mode="align_corners")
        support_f = support_f * mask

    bsz, ch, hs, ws = support_f.shape
    norm = ops.L2Normalize(axis=1, epsilon=1e-12)
    support_f = support_f.view(bsz, ch, -1)
    support_f = support_f / (norm(support_f) + eps)

    bsz, ch, hq, wq = query_f.shape
    query_f = query_f.view(bsz, ch, -1)
    query_f = query_f / (norm(query_f) + eps)

    bmm = ops.BatchMatMul()
    corr = bmm(support_f.transpose((0, 2, 1)), query_f)

    corr = ops.clip_by_value(corr, Tensor(0.0, mindspore.float32), float('inf'))
    
    mean = ops.ReduceMean(keep_dims=True)
    corr = mean(corr, 1).view(bsz, -1, hq, wq)
    
    return corr


class Correlation:
    @classmethod
    def multilayer_correlation(cls, query_feats, support_feats, stack_ids):
        eps = 1e-5
        stack_op = ops.Stack(axis=0)

        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.shape
            support_feat_flat = support_feat.view(bsz, ch, -1)
            support_feat_norm = support_feat_flat / (ops.norm(support_feat_flat, ord=2, dim=1, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.shape
            query_feat_flat = query_feat.view(bsz, ch, -1)
            query_feat_norm = query_feat_flat / (ops.norm(query_feat_flat, ord=2, dim=1, keepdim=True) + eps)


            corr = ops.bmm(query_feat_norm.transpose((0, 2, 1)), support_feat_norm).view(bsz, ha, wa, hb, wb)
            corr = ops.clip_by_value(corr, Tensor(0.0, mindspore.float32), float('inf'))
            corrs.append(corr)

        num_corrs = len(corrs)

        start_l4 = num_corrs - stack_ids[0]
        corr_l4 = stack_op(corrs[start_l4:]).transpose((1, 0, 2, 3, 4, 5))

        start_l3 = num_corrs - stack_ids[1]
        end_l3 = num_corrs - stack_ids[0]
        corr_l3 = stack_op(corrs[start_l3:end_l3]).transpose((1, 0, 2, 3, 4, 5))

        start_l2 = num_corrs - stack_ids[2]
        end_l2 = num_corrs - stack_ids[1]
        corr_l2 = stack_op(corrs[start_l2:end_l2]).transpose((1, 0, 2, 3, 4, 5))

        return [corr_l4, corr_l3, corr_l2]

    @classmethod
    def multi_similarity(cls, query_feats, support_feats, support_mask=None):
        cos_sim = cosine_similarity(query_f=query_feats, support_f=support_feats, support_mask=support_mask)

        return cos_sim
import mindspore
import mindspore.numpy as mnp
import numpy as np


class Evaluator:
    @classmethod
    def initialize(cls):
        cls.ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask, target):
        
        gt_mask = target
        
        B = pred_mask.shape[0]

        pred_mask_flat = pred_mask.reshape(B, -1)
        gt_mask_flat = gt_mask.reshape(B, -1)

        area_inter = mnp.zeros((2, B), mindspore.float32)
        area_pred = mnp.zeros((2, B), mindspore.float32)
        area_gt = mnp.zeros((2, B), mindspore.float32)

        for i in range(B):
            _pred_mask = pred_mask_flat[i]
            _gt_mask = gt_mask_flat[i]
   
            correct_match = (_pred_mask == _gt_mask)
            
            if mnp.sum(correct_match).asnumpy() > 0:

                _inter_values = _gt_mask[correct_match]

                inter_0 = mnp.sum(_inter_values == 0)

                inter_1 = mnp.sum(_inter_values == 1)
                
                area_inter[0, i] = inter_0.astype(mindspore.float32)
                area_inter[1, i] = inter_1.astype(mindspore.float32)
            else:

                area_inter[0, i] = 0.0
                area_inter[1, i] = 0.0
   
            pred_0 = mnp.sum(_pred_mask == 0)
            pred_1 = mnp.sum(_pred_mask == 1)
            area_pred[0, i] = pred_0.astype(mindspore.float32)
            area_pred[1, i] = pred_1.astype(mindspore.float32)

            gt_0 = mnp.sum(_gt_mask == 0)
            gt_1 = mnp.sum(_gt_mask == 1)
            area_gt[0, i] = gt_0.astype(mindspore.float32)
            area_gt[1, i] = gt_1.astype(mindspore.float32)

        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union
import mindspore as ms
from mindspore import nn, ops, Tensor
import numpy as np

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    reduce_sum = ops.ReduceSum()
    epsilon = 1e-6

    for i in range(0,b):

        Iand1 = reduce_sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = reduce_sum(target[i,:,:,:]) + reduce_sum(pred[i,:,:,:])-Iand1

        IoU1 = Iand1/(Ior1 + epsilon)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(nn.Cell):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def construct(self, pred, target):
        return _iou(pred, target, self.size_average)
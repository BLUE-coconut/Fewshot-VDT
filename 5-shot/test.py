import os
import time
import cv2
import numpy as np
import mindspore as ms
from mindspore import nn, ops, context, load_checkpoint, load_param_dict, load_param_into_net
from mindspore.dataset import GeneratorDataset

from common.dataset_mask_train import Tri_Dataset
from model.mymodel import IFCNet_5shot

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=0)  # 假设使用默认设备0

if __name__ == '__main__':

    model_path = './Final/epoch_15.ckpt'

    net = IFCNet_5shot("resnet50", False, dropout=0, shot=5)

    print('loading model from %s...' % model_path)
    net_param_dict = load_checkpoint(model_path)

    load_param_into_net(net, net_param_dict)

    net.set_train(False)

    out_path = './outputs'
    root = 'VDT-2048-5i/'

    dataset_test = Tri_Dataset(
        data_dir=root,
        fold=0,
        normalize_mean=[0.3884923, 0.361114, 0.3357993],
        normalize_std=[0.14982404, 0.1512635, 0.16091296],
        normalize_mean_d=[0.9863242] * 3,
        normalize_std_d=[0.05647239] * 3,
        normalize_mean_th=[0.40243158] * 3,
        normalize_std_th=[0.09522554] * 3,
        mode='test'
    )

    column_names = [
        "input", "input_th", "input_d",
        "s_input", "s_input_th", "s_input_d", "s_mask",
        "H", "W", "name"
    ]
    test_loader = GeneratorDataset(
        dataset_test,
        column_names=column_names,
        shuffle=False,
        num_parallel_workers=1
    )
    test_loader = test_loader.batch(1, drop_remainder=False)

    if not os.path.exists(out_path): os.mkdir(out_path)

    time_s = time.time()
    img_num = test_loader.get_dataset_size()
    i = 0

    for batch in test_loader.create_tuple_iterator():
        (
            inputs, input_th, input_d,
            s_input, s_input_th, s_input_d, s_mask,
            H_tensor, W_tensor, name
        ) = batch

        H = int(H_tensor.asnumpy())
        W = int(W_tensor.asnumpy())

        i += 1
        print(i)

        logit_mask = net(
            inputs.astype(ms.float32),
            input_th.astype(ms.float32),
            input_d.astype(ms.float32),
            s_input.astype(ms.float32),
            s_input_th.astype(ms.float32),
            s_input_d.astype(ms.float32),
            s_mask.astype(ms.float32)
        )

        if isinstance(logit_mask, (tuple, list)):
            masks = logit_mask[-1]
        else:
            masks = logit_mask

        score1 = ops.interpolate(
            masks.astype(ms.float32),
            sizes=(H, W),
            coordinate_transform_mode='align_corners',
            mode=nn.ResizeMode.BILINEAR
        )

        pred_tensor = ops.sigmoid(score1)
        pred = np.squeeze(pred_tensor.asnumpy())

        pred_min = pred.min()
        pred_max = pred.max()
        pred = (pred - pred_min) / (pred_max - pred_min + 1e-8)

        filename = name.asnumpy()[0].decode()
        cv2.imwrite(os.path.join(out_path, filename[:-4] + '.png'), 255 * pred)

    time_e = time.time()
    print('speed: %f FPS' % (img_num / (time_e - time_s)))
import os
import mindspore as ms
from mindspore import nn, ops, Tensor, context, save_checkpoint
from mindspore.train.serialization import save_checkpoint

from common.dataset_mask_train import Tri_Dataset
from mindspore.dataset import GeneratorDataset
from model.mymodel import *

from tqdm import tqdm

bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
L1 = nn.L1Loss(reduction="mean")

def structure_loss(pred, mask):
    avg_pool = ops.AvgPool(kernel_size=31, strides=1, pad_mode="same")(mask)
    diff = avg_pool - mask
    weit = 1 + 5 * diff.abs()

    bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
    bce = bce_criterion(pred, mask)
    wbce = (weit * bce).sum(axis=(2, 3)) / weit.sum(axis=(2, 3))
    
    sigmoid = nn.Sigmoid()
    pred_sigmoid = sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(axis=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(axis=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

class IOULoss(nn.Cell):
    def __init__(self):
        super(IOULoss, self).__init__()

    def construct(self, pred, target):
        pred = ops.sigmoid(pred)
        inter = ops.sum(pred * target, axis=(1, 2, 3))
        union = ops.sum(pred + target, axis=(1, 2, 3)) - inter
        iou = 1 - (inter + 1) / (union + 1)
        return iou.mean()

iou_loss = IOULoss()

if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    img_root = 'VDT-2048-5i/'
    save_path = './Final'
    os.makedirs(save_path, exist_ok=True)

    lr = 1e-5
    batch_size = 1
    epoch = 50

    dataset_train = Tri_Dataset(
        data_dir=img_root,
        fold=0,
        normalize_mean=[0.3884923, 0.361114, 0.3357993],
        normalize_std=[0.14982404, 0.1512635, 0.16091296],
        normalize_mean_d=[0.9863242] * 3,
        normalize_std_d=[0.05647239] * 3,
        normalize_mean_th=[0.40243158] * 3,
        normalize_std_th=[0.09522554] * 3,
    )
    train_loader = GeneratorDataset(
        dataset_train,
        column_names=[
            "input", "input_th", "input_d",
            "target", "s_input", "s_input_th",
            "s_input_d", "s_mask", "subcls"
        ],
        shuffle=True,
        num_parallel_workers=1
    )

    train_loader = train_loader.batch(batch_size, drop_remainder=True)

    net = IFCNet("resnet50", False, dropout=0.2)
    num_params = sum([p.size for p in net.get_parameters()])
    print("The number of parameters: {}".format(num_params))

    net.set_train()
    unused_params_names = [
    'backbone.head.dense.weight', 
    'backbone.head.dense.bias'
    ]

    trainable_params = []
    for param in net.trainable_params():
        if param.name not in unused_params_names:
            trainable_params.append(param)
    optimizer = nn.Adam(trainable_params, learning_rate=lr, beta1=0.5, beta2=0.999)

    def forward_fn(inputs, input_th, input_d, s_input, s_input_th, s_input_d, s_mask, target):
        logit_mask = net(inputs, input_th, input_d, s_input, s_input_th, s_input_d, s_mask)
        
        if isinstance(logit_mask, (tuple, list)):
            logit_mask = logit_mask[-1]
            
        target_unsq = ops.expand_dims(target, 1)
        loss = structure_loss(logit_mask.astype(ms.float32), target_unsq.astype(ms.float32))
        return loss

    grad_op = ops.GradOperation(get_by_list=True)
    grad_fn = grad_op(forward_fn, optimizer.parameters)
    steps_per_epoch = train_loader.get_dataset_size() 

    for epochi in range(1, epoch + 1):

        tqdm_bar = tqdm(
            train_loader.create_tuple_iterator(), 
            total=steps_per_epoch,
            desc=f"Epoch {epochi}/{epoch}" 
        )

        for batch_idx, batch in enumerate(tqdm_bar):
            (
                inputs, input_th, input_d,
                target, s_input, s_input_th,
                s_input_d, s_mask, subcls
            ) = batch

            target = target.squeeze(1)
            s_mask = s_mask

            loss = forward_fn(inputs, input_th, input_d, s_input, s_input_th, s_input_d, s_mask, target)
            grads = grad_fn(inputs, input_th, input_d, s_input, s_input_th, s_input_d, s_mask, target)
            optimizer(grads)
            tqdm_bar.set_postfix(Loss=f"{loss.asnumpy():.6f}")

        print(f"Epoch {epochi}/{epoch} completed. Last Batch Loss = {loss.asnumpy():.6f}")

        if epochi % 19 == 0:
            save_checkpoint(net, f"{save_path}/epoch_{epochi}.ckpt")

    save_checkpoint(net, f"{save_path}/final.ckpt")
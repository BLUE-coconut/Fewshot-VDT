r""" Hypercorrelation Squeeze training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch

from model.asnet import *
from model.hsnet import *
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.vis import Visualizer
from common import utils
from data.dataset import FSSDataset
import os
from common import dataset_mask_train, dataset_mask_val

########## use Tri_Dataset #########


def trainv2(epoch, model, dataloader, optimizer, sub_list, training, auxloss_weight = 0.3):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    #utils.fix_randseed(0) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(sub_list)

    for idx, (input, _, _, target, s_input, _, _, s_mask, subcls) in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        s_mask = s_mask.squeeze(1)
        target = target.squeeze(1)
        s_input = s_input.cuda(non_blocking=True)
        #s_input_d = s_input_d.cuda(non_blocking=True)
        #s_input_th = s_input_th.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        #input_d = input_d.cuda(non_blocking=True)
        #input_th = input_th.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        subcls = subcls.cuda(non_blocking=True)
        logit_mask = model(input, s_input, s_mask)
        if isinstance(logit_mask,list):
            logit_mask = logit_mask[-1]
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.module.compute_objective(logit_mask, target)
        loss_sum = loss

        if training:
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, target)
        average_meter.update(area_inter, area_union, subcls, loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou

def main(model):
    # Model initialization
    if args.fold == 3:
        sub_list = list(range(0, 15))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        sub_val_list = list(range(15, 20))  # [16,17,18,19,20]
    elif args.fold == 2:
        sub_list = list(range(0, 10)) + list(range(15, 20))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(10, 15))  # [6,7,8,9,10]
    elif args.fold == 1:
        sub_list = list(range(0, 5)) + list(range(10, 20))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(5, 10))
    elif args.fold == 0:
        sub_list = list(range(5, 20))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        sub_val_list = list(range(0, 5))
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    model = nn.DataParallel(model)
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    Evaluator.initialize()
    start_epoch = 0

    # Dataset initialization
    FSSDataset.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=False)


    if(args.resume is not None):
        model.load_state_dict(torch.load(os.path.join(args.resume, 'best_model.pt'), map_location=device),strict=False)

    dataloader_trn = dataset_mask_train.Tri_Dataset(data_dir=args.datapath, fold=args.fold,
                                                  normalize_mean=[0.3884923,0.361114,0.3357993], normalize_std=[0.14982404,0.1512635,0.16091296],
                                                  normalize_mean_d=[0.9863242,0.9863242,0.9863242], normalize_std_d=[0.05647239,0.05647239,0.05647239],
                                                  normalize_mean_th=[0.40243158,0.40243158,0.40243158], normalize_std_th=[0.09522554,0.09522554,0.09522554])
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataloader_trn, batch_size=args.bsz, shuffle=(train_sampler is None),
                                               num_workers=args.nworker, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)

    dataloader_val = dataset_mask_val.Tri_Dataset(data_dir=args.datapath, fold=args.fold,
                                                  normalize_mean=[0.3884923,0.361114,0.3357993], normalize_std=[0.14982404,0.1512635,0.16091296],
                                                  normalize_mean_d=[0.9863242,0.9863242,0.9863242], normalize_std_d=[0.05647239,0.05647239,0.05647239],
                                                  normalize_mean_th=[0.40243158,0.40243158,0.40243158], normalize_std_th=[0.09522554,0.09522554,0.09522554])
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(dataloader_val, batch_size=args.bsz, shuffle=False,
                                             num_workers=args.nworker, pin_memory=True, sampler=val_sampler)

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_fb_miou = float('-inf')
    best_val_loss = float('inf')
    best_epoch = float('inf')
    for epoch in range(start_epoch,args.epoch):
        trn_loss, trn_miou, trn_fb_iou = trainv2(epoch, model, train_loader, optimizer, sub_list, training=True, auxloss_weight = 0.3)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = trainv2(epoch, model, val_loader, optimizer, sub_val_list, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_val_fb_miou = val_fb_iou
            best_epoch = epoch
            Logger.save_model_miou(model, best_epoch, val_miou)

        print("%d epoch ,%.2f is the best M-iou, %.2f is the best FB-iou" % (best_epoch, best_val_miou, best_val_fb_miou))

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='./VDT-2048-5i') #yes
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='') 
    parser.add_argument('--bsz', type=int, default=2) #yes
    parser.add_argument('--lr', type=float, default=1e-4) #yes
    parser.add_argument('--epoch', type=int, default=200) #yes
    parser.add_argument('--nworker', type=int, default=2) #yes
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3]) #yes
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101']) #yes
    parser.add_argument('--model', type=str, default='v1')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    if(args.model == 'hsnet'):
        Logger.initialize(args, training=True, modelname="HypercorrSqueezeNetwork")
        model = HypercorrSqueezeNetwork(args.backbone, False)
    elif(args.model == 'asnet'):
        Logger.initialize(args, training=True, modelname="AttentiveSqueezeNetwork")
        model = AttentiveSqueezeNetwork(args, False)
    
    else:
        raise KeyError(f"model version {args.model} not defined")

    

    Logger.log_params(model)
    main(model)



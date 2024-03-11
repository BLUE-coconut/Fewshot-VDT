r""" Hypercorrelation Squeeze training (validation) code """
import argparse

import torch.optim as optim
import torch.nn as nn
import torch
from model.mymodel import *
from model.ab_model import *
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
import os
from common import dataset_mask_train, dataset_mask_val
from common.vis import Visualizer
import cv2
import numpy as np
import PIL.Image as Image
import os
from os.path import join
#import matplotlib.pyplot as plt


def normPRED(d):
    ma = np.max(d)
    mi = np.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def get_colormap(ori,predict,save_path):
    #print("input",ori.shape)
    #print("predict",predict.shape)
    if(not isinstance(predict,np.ndarray)):
        predict = predict.cpu().data.numpy()
    if(len(predict.shape)==4 and predict.shape[1]!=1):
        predict = np.mean(predict,axis=1)
    predict = normPRED(predict)
    predict = predict.squeeze()*255
    predict = np.clip(predict,0,255).astype(np.uint8)
    

    if(isinstance(ori,torch.Tensor)):
        ori = ori.cpu().data.numpy()
    elif(isinstance(ori,Image.Image)):
        ori = np.array(ori)
    ori = ori.squeeze()*255
    #print("ori",ori.shape)
    ori = ori.astype(np.uint8).transpose((1,2,0))
    #print("input2",ori.shape)
    #print("predict2",predict.shape)

    im = cv2.applyColorMap(predict, cv2.COLORMAP_JET)
    #print("im",im.shape)
    
    heat = cv2.resize(im,(ori.shape[0],ori.shape[1]),interpolation=cv2.INTER_LINEAR)
    #print("heat",heat.shape)
    #ori = cv2.cvtColor(ori[:,:,None],cv2.COLOR_GRAY2BGR)
    #print(heat.shape,ori.shape)
    
    img_heat = cv2.addWeighted(ori, 0.5, heat, 0.5, 0)
    cv2.imwrite(save_path, img_heat)
    #print(save_path)

def visaulize(model, dataloader, sub_list, nshot=1, num_pick=15):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)
    average_meter = AverageMeter(sub_list)

    # for idx, (input, input_th, input_d, target, s_input, s_input_th, s_input_d, s_mask, subcls, label, slabel,paths_rgb,paths_d,paths_th,pathq_rgb,pathq_d,pathq_th) in enumerate(dataloader):
    for idx, (input, input_th, input_d, target, s_input, s_input_th, s_input_d, s_mask, subcls, img_name, ori) in enumerate(dataloader):
        if(idx>=num_pick):break
        print(f"inference {idx}")

        # 1. Hypercorrelation Squeeze Networks forward pass

        s_mask =s_mask.squeeze(1)
        target = target.squeeze(1)
        s_input = s_input.cuda(non_blocking=True)
        s_input_d = s_input_d.cuda(non_blocking=True)
        s_input_th = s_input_th.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        input_d = input_d.cuda(non_blocking=True)
        input_th = input_th.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        subcls = subcls.cuda(non_blocking=True)
        if(args.visualize):
            logit_masks,vsim_maps,dsim_maps,tsim_maps,multi_sim,vis_l3,vis_l4 = model(input,input_th,input_d, s_input,s_input_th,s_input_d, s_mask)
        else:
            logit_masks = model(input,input_th,input_d, s_input,s_input_th,s_input_d, s_mask)

        pred_mask = logit_masks[-1].argmax(dim=1)
        

        if(args.visualize):
            
            img_name = img_name[0]
            input = ori
            
            save_root = join("./SEMA-1shot/output",args.load.split("/")[2].split(".")[0],img_name.split(".")[0])
            if not os.path.exists(save_root):
                os.makedirs(save_root, exist_ok=True)

            if not os.path.exists(join(save_root,"multi_vsim")):
                os.makedirs(join(save_root,"multi_vsim"), exist_ok=True)
                
            if not os.path.exists(join(save_root,"multi_dsim")):
                os.makedirs(join(save_root,"multi_dsim"), exist_ok=True)

            if not os.path.exists(join(save_root,"multi_tsim")):
                os.makedirs(join(save_root,"multi_tsim"), exist_ok=True)

            if not os.path.exists(join(save_root,"multi_trisim")):
                os.makedirs(join(save_root,"multi_trisim"), exist_ok=True)

            if not os.path.exists(join(save_root,"Tricorr_l3")):
                os.makedirs(join(save_root,"Tricorr_l3"), exist_ok=True)

            if not os.path.exists(join(save_root,"Tricorr_l4")):
                os.makedirs(join(save_root,"Tricorr_l4"), exist_ok=True)

            if not os.path.exists(join(save_root,"pred_out")):
                os.makedirs(join(save_root,"pred_out"), exist_ok=True)
            
            
            for idx,vsims in enumerate(vsim_maps):
                for iidx,vsim in enumerate(vsims):
                    save_file = join(save_root,"multi_vsim",str(idx)+"_"+str(iidx)+"_"+img_name)
                    get_colormap(input,vsim,save_file)
            for idx,dsims in enumerate(dsim_maps):
                for iidx,dsim in enumerate(dsims):
                    save_file = join(save_root,"multi_dsim",str(idx)+"_"+str(iidx)+"_"+img_name)
                    get_colormap(input,dsim,save_file)
            for idx,tsims in enumerate(tsim_maps):
                for iidx,tsim in enumerate(tsims):
                    save_file = join(save_root,"multi_tsim",str(idx)+"_"+str(iidx)+"_"+img_name)
                    get_colormap(input,tsim,save_file)
            for idx,msim in enumerate(multi_sim):
                save_file = join(save_root,"multi_trisim",str(idx)+"_"+img_name)
                get_colormap(input,msim,save_file)

            save_file = join(save_root,"Tricorr_l3",img_name)
            get_colormap(input,vis_l3[-1],save_file)
            save_file = join(save_root,"Tricorr_l4",img_name)
            get_colormap(input,vis_l4[-1],save_file)

            #get_colormap(input,multi_corr[0].mean(dim=1),join(save_root,img_name.split(".")[0]+"tri_l4.png"))
            
        
            #pred_mask_np = pred_mask.cpu().data.numpy()
            #print(pred_mask.max())
            #pred_mask_np = np.clip(pred_mask_np*255,0,255).astype(np.uint8)
            #cv2.imwrite(join(save_root,"pred_out",img_name), pred_mask)
            get_colormap(input,pred_mask,join(save_root,"pred_out",img_name.split(".")[-2]+"_color"+".png"))
            print(pred_mask.shape)
            img = Image.fromarray((pred_mask.squeeze(0).cpu().data.numpy() * 255).astype(np.uint8))  
            # 保存图像为 PNG 文件  
            img.save(join(save_root,"pred_out",img_name))

            #pred_mask_tr = cv2.cvtColor(pred_mask.cpu().data.numpy().transpose((1,2,0)),cv2.COLOR_GRAY2RGB)
            #cv2.imwrite(join(save_root,"pred_out",img_name), pred_mask_tr)

        #visulize(paths_rgb,paths_d,paths_th,pathq_rgb,pathq_d,pathq_th, label.cpu(), slabel.cpu(),pred_mask.cpu(),idx)
        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, target)
        average_meter.update(area_inter, area_union, subcls, None)
        average_meter.write_process(idx, len(dataloader), -1, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou

def trainv2(epoch, model, dataloader, optimizer,sub_list, training, auxloss_weight = 0.3):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    #utils.fix_randseed(None) if training else utils.fix_randseed(0)
    utils.fix_randseed(0) if training else utils.fix_randseed(0)
    model.module.train_mode() if training else model.module.eval()
    average_meter = AverageMeter(sub_list)

    for idx, (input, input_th, input_d, target, s_input, s_input_th, s_input_d, s_mask, subcls) in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        s_mask = s_mask.squeeze(1)
        target = target.squeeze(1)
        s_input = s_input.cuda(non_blocking=True)
        s_input_d = s_input_d.cuda(non_blocking=True)
        s_input_th = s_input_th.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        input_d = input_d.cuda(non_blocking=True)
        input_th = input_th.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        subcls = subcls.cuda(non_blocking=True)
        logit_mask = model(input,input_th,input_d, s_input,s_input_th,s_input_d, s_mask)
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
    parser.add_argument('--datapath', type=str, default='./VDT-2048-5i')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--load', type=str, default='./logs/_0102_084221_HS_VDT_v2_10.log/')
    parser.add_argument('--model', type=str, default='v10')

    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    if(args.model == 'v10'):
        Logger.initialize(args, training=False, modelname="HS_VDT_v2_10")
        model = HS_VDT_v2_10(args.backbone, False, dropout=args.drop)
    elif(args.model == '345-345'):
        Logger.initialize(args, training=False, modelname="MCS345l_TC345l")
        model = MCS345l_TC345l(args.backbone, False, dropout=args.drop)
    elif(args.model == '45-45'):
        Logger.initialize(args, training=False, modelname="MCS45l_TC45l")
        model = MCS45l_TC45l(args.backbone, False, dropout=args.drop)
    elif(args.model == '345pro-345'):
        Logger.initialize(args, training=False, modelname="MCSpro345l_TC345l")
        model = MCSpro345l_TC345l(args.backbone, False, dropout=args.drop)
    elif(args.model == '345pro-45'):
        Logger.initialize(args, training=False, modelname="MCSpro345l_TC45l")
        model = MCSpro345l_TC45l(args.backbone, False, dropout=args.drop)    
    elif(args.model == 'wo-345'):
        Logger.initialize(args, training=False, modelname="MCSwo_TC345l")
        model = MCSwo_TC345l(args.backbone, False, dropout=args.drop)
    elif(args.model == 'wo-45'):
        Logger.initialize(args, training=False, modelname="MCSwo_TC45l")
        model = MCSwo_TC45l(args.backbone, False, dropout=args.drop)
    elif(args.model == '345-wo'):
        Logger.initialize(args, training=False, modelname="MCS345l_TCwo")
        model = MCS345l_TCwo(args.backbone, False, dropout=args.drop)  
    elif(args.model == '345-45-D'):
        Logger.initialize(args, training=False, modelname="MCS345l_TC45l_D")
        model = MCS345l_TC45l_D(args.backbone, False, dropout=args.drop) 
    elif(args.model == 'wo-45-D'):
        Logger.initialize(args, training=False, modelname="MCSwo_TC45l_D")
        model = MCSwo_TC45l_D(args.backbone, False, dropout=args.drop)  
    elif(args.model == '345-45-T'):
        Logger.initialize(args, training=False, modelname="MCS345l_TC45l_T")
        model = MCS345l_TC45l_T(args.backbone, False, dropout=args.drop)  

    else:
        raise KeyError(f"model version {args.model} not defined")
    
    model.eval()
    Logger.log_params(model)

    main(model)
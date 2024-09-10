r""" Hypercorrelation Squeeze testing code """
import argparse
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2
from model.ab_model import *
from model.mymodel import *
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset
from common import dataset_mask_train, dataset_mask_val
import os
from tqdm import tqdm
from PIL import Image

def masking(img, mask, color=[0,0,255], alph=0.5):
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask == 255] = color
    out = cv2.addWeighted(img_layer, alph, out, 1-alph, 0, out)
    return out

def visual_depth(img):
    img = img.astype(np.float64)
    min_val = np.min(img)
    max_val = np.max(img)
    img_norm = (img - min_val) / (max_val - min_val)
    #print(img_norm)
    img_norm = (img_norm*255).astype(np.uint8)
    #print(img_norm)
    return img_norm

def apply_mask(image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """
        for c in range(image.shape[2]):
            image[:, :, c] = np.where(mask >0.5*255,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])
        return image


def visulize(ori_query_rgb,ori_query_d,ori_query_th,q_mask, ori_support_rgb,ori_support_d,ori_support_th,s_mask, ori_name, q_pred,i):
    # 綠色：[0,255,0]  黃色
    # save query image
    #print(q_mask.shape)
    colors_dict = {'red': (255, 50, 50), 'blue': (102, 140, 255), 'green':(0,255,0)}

    #print(q_mask[q_mask>0])

    ori_query_rgb = ori_query_rgb.cpu().squeeze().numpy()
    ori_query_d = ori_query_d.cpu().squeeze().numpy()
    ori_query_th = ori_query_th.cpu().squeeze().numpy()
    q_mask = q_mask.cpu().squeeze().numpy()

    ori_support_rgb = ori_support_rgb.cpu().squeeze().numpy()
    ori_support_d = ori_support_d.cpu().squeeze().numpy()
    ori_support_th = ori_support_th.cpu().squeeze().numpy()
    s_mask = s_mask.cpu().squeeze().numpy()

    #print(q_mask.shape,ori_query_rgb.shape,np.max(ori_query_rgb),np.max(q_mask))   # (480, 640),(480, 640, 3)
    q_h_ori = q_mask.shape[0]
    q_w_ori = q_mask.shape[1]

    q_pred = F.interpolate(q_pred.float().unsqueeze(1), [q_h_ori, q_w_ori], mode='bilinear', align_corners=True)#[1,h,w]
    q_pred = q_pred.cpu().squeeze().numpy()*255 # [h,w]

    
    path1 = args.visdir

    query_gt = Image.fromarray(apply_mask(ori_query_rgb.copy(),q_mask,colors_dict['blue']))
    query_gt.save(os.path.join(path1,ori_name+'_query_GT.png'))

    query_rgb = Image.fromarray(apply_mask(ori_query_rgb.copy(),q_pred,colors_dict['red']))
    query_rgb.save(os.path.join(path1,ori_name+'_query_pred.png'))

    support_gt = Image.fromarray(apply_mask(ori_support_rgb.copy(),s_mask,colors_dict['green']))
    support_gt.save(os.path.join(path1,ori_name+'_support_GT.png'))

    ori_query_rgb = Image.fromarray(ori_query_rgb)
    ori_query_rgb.save(os.path.join(path1,ori_name+'_query_rgb.png'))

    ori_query_d = Image.fromarray(visual_depth(ori_query_d))
    ori_query_d.save(os.path.join(path1,ori_name+'_query_d.png'))

    ori_query_th = Image.fromarray(ori_query_th)
    ori_query_th.save(os.path.join(path1,ori_name+'_query_th.png'))

    ori_support_rgb = Image.fromarray(ori_support_rgb)
    ori_support_rgb.save(os.path.join(path1,ori_name+'_support_rgb.png'))

    ori_support_d = Image.fromarray(visual_depth(ori_support_d))
    ori_support_d.save(os.path.join(path1,ori_name+'_support_d.png'))

    ori_support_th = Image.fromarray(ori_support_th)
    ori_support_th.save(os.path.join(path1,ori_name+'_support_th.png'))

    #cv2.imwrite(os.path.join(path1,'query_V.png'), ori_query_rgb)
    #cv2.imwrite(os.path.join(path1,'query_GT.png'), q_mask_rgb)
    


def test(model, dataloader, sub_list):
    r""" Test HSNet """

    average_meter = AverageMeter(sub_list)

    # query_rgb, query_th, query_d, query_mask.long(), support_rgb,support_th, support_d, support_mask.long(), sample_class-1, image_path_maskq,image_path_vq,image_path_dq,image_path_thq, image_path_masks,image_path_vs,image_path_ds,image_path_ths
        
    if(args.mode=='vis'):
        for idx, (query_rgb, query_th, query_d, query_mask, support_rgb, support_th, support_d, support_mask, subcls, ori_query_rgb, ori_query_th, ori_query_d, ori_query_mask, ori_support_rgb,ori_support_th, ori_support_d, ori_support_mask, ori_name) in enumerate(tqdm(dataloader)):
                # 1. Hypercorrelation Squeeze Networks forward pass
            
            support_mask =support_mask.squeeze(1)
            query_mask = query_mask.squeeze(1)
            support_rgb = support_rgb.cuda(non_blocking=True)
            support_d = support_d.cuda(non_blocking=True)
            support_th = support_th.cuda(non_blocking=True)
            support_mask = support_mask.cuda(non_blocking=True)
            query_rgb = query_rgb.cuda(non_blocking=True)
            query_d = query_d.cuda(non_blocking=True)
            query_th = query_th.cuda(non_blocking=True)
            query_mask = query_mask.cuda(non_blocking=True)
            subcls = subcls.cuda(non_blocking=True)
            logit_mask = model(query_rgb,query_th,query_d, support_rgb,support_th,support_d, support_mask)
            if isinstance(logit_mask,list):
                logit_mask = logit_mask[-1]
            pred_mask = logit_mask.argmax(dim=1)

            loss = model.module.compute_objective(logit_mask, query_mask)


            if(idx%10==0):visulize(ori_query_rgb,ori_query_d,ori_query_th,ori_query_mask, ori_support_rgb,ori_support_d,ori_support_th,ori_support_mask, ori_name[0], pred_mask,idx)
            # 2. Evaluate prediction
            area_inter, area_union = Evaluator.classify_prediction(pred_mask, query_mask)
            average_meter.update(area_inter, area_union, subcls, loss.detach().clone())



        # Write evaluation results
        
        average_meter.write_result('Test', 0)
        miou, fb_iou = average_meter.compute_iou()
        return miou, fb_iou
    else:
        for idx, (query_rgb, query_th, query_d, query_mask, support_rgb, support_th, support_d, support_mask, subcls) in enumerate(tqdm(dataloader)):

            # 1. Hypercorrelation Squeeze Networks forward pass

            support_mask =support_mask.squeeze(1)
            query_mask = query_mask.squeeze(1)
            support_rgb = support_rgb.cuda(non_blocking=True)
            support_d = support_d.cuda(non_blocking=True)
            support_th = support_th.cuda(non_blocking=True)
            support_mask = support_mask.cuda(non_blocking=True)
            query_rgb = query_rgb.cuda(non_blocking=True)
            query_d = query_d.cuda(non_blocking=True)
            query_th = query_th.cuda(non_blocking=True)
            query_mask = query_mask.cuda(non_blocking=True)
            subcls = subcls.cuda(non_blocking=True)
            logit_mask = model(query_rgb,query_th,query_d, support_rgb,support_th,support_d, support_mask)
            if isinstance(logit_mask,list):
                logit_mask = logit_mask[-1]
            pred_mask = logit_mask.argmax(dim=1)

            loss = model.module.compute_objective(logit_mask, query_mask)


            # 2. Evaluate prediction
            area_inter, area_union = Evaluator.classify_prediction(pred_mask, query_mask)
            average_meter.update(area_inter, area_union, subcls, loss.detach().clone())



        # Write evaluation results
        
        average_meter.write_result('Test', 0)
        miou, fb_iou = average_meter.compute_iou()
        return miou, fb_iou

def main(model):
    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)


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
        
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(os.path.join(args.load,"best_model.pt")))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    #Visualizer.initialize(args.visualize)

    # Dataset initialization

    dataloader_val = dataset_mask_val.Tri_Dataset_test(data_dir=args.datapath, fold=args.fold, shot=args.shot,
                                                  normalize_mean=[0.3884923,0.361114,0.3357993], normalize_std=[0.14982404,0.1512635,0.16091296],
                                                  normalize_mean_d=[0.9863242,0.9863242,0.9863242], normalize_std_d=[0.05647239,0.05647239,0.05647239],
                                                  normalize_mean_th=[0.40243158,0.40243158,0.40243158], normalize_std_th=[0.09522554,0.09522554,0.09522554],
                                                  mode=args.mode)
    
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(dataloader_val, batch_size=1, shuffle=False,
                                             num_workers=args.nworker, pin_memory=True, sampler=val_sampler)

    # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, val_loader, sub_val_list)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')




if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../SEMA/VDT-2048-5i')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=2)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='1-shot/saved_model/FC-D/fold0')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--model', type=str, default='v1')
    parser.add_argument('--visdir', type=str, default="./visual_maps")
    args = parser.parse_args()

    if (args.mode == 'vis' and not os.path.exists(args.visdir)):
            os.mkdir(args.visdir)
    
    if(args.model == 'v10'):
        Logger.initialize(args, training=True, modelname="HS_VDT_v2_10")
        model = HS_VDT_v2_10(args.backbone, False)
    elif(args.model == '345-45-Vm'):
        Logger.initialize(args, training=True, modelname="HS_VDT_v2_10_4_mean")
        model = HS_VDT_v2_10_4_mean(args.backbone, False, shot=args.shot)
    elif(args.model == '345-345m'):
        Logger.initialize(args, training=True, modelname="MCS345l_TC345l_mean")
        model = MCS345l_TC345l_mean(args.backbone, False, shot=args.shot)
    elif(args.model == '45-45m'):
        Logger.initialize(args, training=True, modelname="MCS45l_TC45l_mean")
        model = MCS45l_TC45l_mean(args.backbone, False, shot=args.shot)
    elif(args.model == '345pro-345m'):
        Logger.initialize(args, training=True, modelname="MCSpro345l_TC45l_mean")
        model = MCSpro345l_TC45l_mean(args.backbone, False, shot=args.shot)
    elif(args.model == 'wo-345m'):
        Logger.initialize(args, training=True, modelname="MCSwo_TC345l_mean")
        model = MCSwo_TC345l_mean(args.backbone, False, shot=args.shot)
    elif(args.model == 'wo-45m'):
        Logger.initialize(args, training=True, modelname="MCSwo_TC45l_mean")
        model = MCSwo_TC45l_mean(args.backbone, False, shot=args.shot)
    elif(args.model == '345-wom'):
        Logger.initialize(args, training=True, modelname="MCS345l_TCwo_mean")
        model = MCS345l_TCwo_mean(args.backbone, False, shot=args.shot)
    elif(args.model == '345-45-Dm'):
        Logger.initialize(args, training=True, modelname="MCS345l_TC45l_D_mean")
        model = MCS345l_TC45l_D_mean(args.backbone, False, shot=args.shot)
    elif(args.model == 'wo-45-Dm'):
        Logger.initialize(args, training=True, modelname="MCSwo_TC45l_D_mean")
        model = MCSwo_TC45l_D_mean(args.backbone, False, shot=args.shot)
    elif(args.model == '345-45-Tm'):
        Logger.initialize(args, training=True, modelname="MCS345l_TC45l_T_mean")
        model = MCS345l_TC45l_T_mean(args.backbone, False, shot=args.shot)
    
    else:
        raise KeyError("model version not defined")

    # Model initialization


    main(model)

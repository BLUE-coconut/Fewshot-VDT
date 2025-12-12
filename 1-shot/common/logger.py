import datetime
import logging
import os

from mindspore.train.summary import SummaryRecord
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor


class AverageMeter:
    def __init__(self, class_ids_interest):
        self.benchmark = 'pascal'

        self.class_ids_interest = Tensor(class_ids_interest, mindspore.int32)

        if self.benchmark == 'pascal':
            self.nclass = 20
        elif self.benchmark == 'coco':
            self.nclass = 80
        elif self.benchmark == 'fss':
            self.nclass = 1000

        self.intersection_buf = mnp.zeros((2, self.nclass), mindspore.float32)
        self.union_buf = mnp.zeros((2, self.nclass), mindspore.float32)

        self.ones = mnp.ones_like(self.union_buf)
        
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id, loss):

        inter_b = Tensor(inter_b, mindspore.float32)
        union_b = Tensor(union_b, mindspore.float32)
        class_id = Tensor(class_id, mindspore.int32)

        for i in range(class_id.shape[0]):
            c_id = class_id[i].asnumpy().item() 
            self.intersection_buf[:, c_id] += inter_b[:, i]
            self.union_buf[:, c_id] += union_b[:, i]

        if loss is None:
            loss = Tensor(0.0, mindspore.float32)
        else:
            loss = Tensor(loss, mindspore.float32)
            
        self.loss_buf.append(loss)

    def compute_iou(self):

        stacked_union = mnp.stack([self.union_buf, self.ones], axis=0)
        union_max = mnp.max(stacked_union, axis=0) 
        iou = self.intersection_buf / union_max

        iou = iou[:, self.class_ids_interest.asnumpy()] 

        miou = mnp.mean(iou[1, :]) * 100

        fb_in = self.intersection_buf[:, self.class_ids_interest.asnumpy()].sum(axis=1)
        fb_un = self.union_buf[:, self.class_ids_interest.asnumpy()].sum(axis=1)

        fb_iou = mnp.mean(fb_in / fb_un) * 100

        return miou.asnumpy().item(), fb_iou.asnumpy().item()

    def write_result(self, split, epoch):
        iou, fb_iou = self.compute_iou()

        loss_buf = mnp.stack(self.loss_buf)
        avg_loss = mnp.mean(loss_buf).asnumpy().item()
        
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch
        msg += 'Avg L: %6.5f  ' % avg_loss
        msg += 'mIoU: %5.2f  ' % iou
        msg += 'FB-IoU: %5.2f  ' % fb_iou

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=2):
        if (batch_idx + 1) % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx + 1, datalen)
            iou, fb_iou = self.compute_iou()
            if epoch != -1:
                loss_buf = mnp.stack(self.loss_buf)
                last_loss = loss_buf[-1].asnumpy().item()
                avg_loss = mnp.mean(loss_buf).asnumpy().item()
                msg += 'L: %6.5f  ' % last_loss
                msg += 'Avg L: %6.5f  ' % avg_loss
            msg += 'mIoU: %5.2f  |  ' % iou
            msg += 'FB-IoU: %5.2f' % fb_iou
            Logger.info(msg)


class Logger:
    logpath = None
    tbd_writer = None
    benchmark = None
    
    @classmethod
    def initialize(cls, args, training, modelname = None):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')

        if training:
            logpath = args.logpath
        else:
            load_path_parts = args.load.split('/')
            test_info = load_path_parts[-2].split('.')[0] if len(load_path_parts) >= 2 else "unknown_load"
            logpath = '_TEST_' + test_info + logtime
            
        if logpath == '': 
            logpath = logtime
            
        if modelname is None:
            modelname = "HSNet"

        cls.logpath_base = os.path.join('logs', logpath)
        cls.logpath = os.path.join(cls.logpath_base, modelname + logtime)
        cls.benchmark = args.benchmark

        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        cls.tbd_writer = SummaryRecord(os.path.join(cls.logpath, 'tbd/runs')) 

        logging.info("\n:=========== Few-shot Seg. with "+modelname+" ===========")
        
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':================================================\n')

    @classmethod
    def info(cls, msg):
        logging.info(msg)
        

    @classmethod
    def save_model_miou(cls, model, epoch, val_miou):
        save_path = os.path.join(cls.logpath, 'best_model.ckpt')
        mindspore.save_checkpoint(model, save_path)
        cls.info('Model saved @%d w/ val. mIoU: %5.2f.\n' % (epoch, val_miou))
    
    @classmethod
    def save_ckp(cls, optimizer, epoch):
        checkpoint = {
            'optimizer': optimizer.parameters_dict(),
            "epoch": epoch
        }
        save_path = os.path.join(cls.logpath, 'best_ckp.ckpt')
        mindspore.save_checkpoint(checkpoint, save_path)
        cls.info('Checkpoint saved @%d' % (epoch))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0

        for param in model.get_parameters():
            k = param.name

            n_param = param.size
            
            if k.split('.')[0] in 'backbone':
                if len(k.split('.')) > 1 and k.split('.')[1] in ['classifier', 'fc']:
                    continue
                backbone_param += n_param
            else:
                learner_param += n_param
                
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % (backbone_param + learner_param))
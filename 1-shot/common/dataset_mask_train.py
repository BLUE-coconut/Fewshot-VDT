import cv2
import random
import os
import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time



class Dataset(object):

    def __init__(self, data_dir, fold, input_size=[400, 400] , normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225],prob=0.7):
        # -------------------load data list,[class,video_name]-------------------
        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        self.data_enhancement(tuple(input_size))
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.prob = prob  # probability of sampling history masks=0
        self.split = fold

        if self.split == 3:
            self.sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        elif self.split == 2:
            self.sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
        elif self.split == 1:
            self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        elif self.split == 0:
            self.sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []
        fold_list = [0, 1, 2, 3]
        fold_list.remove(fold)
        for fold in fold_list:
            f = open(os.path.join(self.data_dir, 'Binary_map','split%1d.txt'%fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                item2 = item.split("_")
                img_name = item2[0]
                cat = int(item2[1])
                new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def data_enhancement(self,img_size):
        self.transs = transforms.Compose([
                            transforms.RandomRotation((-10, 10)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(img_size),
                            ])
        self.transq = transforms.Compose([
                            transforms.RandomRotation((-10, 10)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(img_size,ratio=(0.9,1.1)),
                            ])

    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = transforms.ToTensor()
        self.resize = transforms.Resize(input_size)
        self.normalize = transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 21):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):

        # give an query index,sample a target class first
        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]  # random sample a class in this img

        support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
        while True:  # random sample a support data
            support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                break

        # print (query_name,support_name)
        support_name_rgb= support_name
        support_name_rgb = support_name_rgb.replace('.png','_rgb.png')########################3
        support_name_th = support_name.replace('.png','_th.png')
        support_name_d = support_name.replace('.png', '_d.png')
        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(random.uniform(1, 1.5)*input_size)
        scale_transform_mask = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_d = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = random.random()

        # image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        # image_th = Image.fromarray(cv2.cvtColor(image_th,cv2.COLOR_BGR2RGB))
        image_th = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        image_d = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_d)).convert('RGB')

        support_th = self.normalize(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag,image_th))))

        support_d = self.normalize(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag,image_d))))

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', support_name_rgb))))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           support_name )))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)
        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_d = support_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        # random scale and crop for query
        scaled_size = input_size  # random.randint(323, 350)
        scale_transform_d = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_th = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_mask = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0  # random.random()'
        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png','_rgb.png')######
        query_name_th = query_name.replace('.png','_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        # image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        # image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))
        image_thq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        image_dq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_d)).convert('RGB')

        query_th = self.normalize(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag,image_thq))))

        query_d = self.normalize(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag,image_dq))))

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', query_name_rgb ))))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           query_name)))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        return query_rgb, query_th, query_d, query_mask.long(), support_rgb, support_th, support_d, support_mask.long(), sample_class-1

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return len(self.new_exist_class_list)


class Dataset_0(object):

    def __init__(self, data_dir, fold, input_size=[400, 400] , normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225],prob=0.7):
        # -------------------load data list,[class,video_name]-------------------
        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.prob = prob  # probability of sampling history masks=0
        self.split = fold

        if self.split == 3:
            self.sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        elif self.split == 2:
            self.sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
        elif self.split == 1:
            self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        elif self.split == 0:
            self.sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []
        fold_list = [0, 1, 2, 3]
        fold_list.remove(fold)
        for fold in fold_list:
            f = open(os.path.join(self.data_dir, 'Binary_map','split%1d.txt'%fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                item2 = item.split("_")
                img_name = item2[0]
                cat = int(item2[1])
                new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = transforms.ToTensor()
        self.resize = transforms.Resize(input_size)
        self.normalize = transforms.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 21):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):

        # give an query index,sample a target class first
        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]  # random sample a class in this img

        support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
        while True:  # random sample a support data
            support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                break

        # print (query_name,support_name)
        support_name_rgb= support_name
        support_name_rgb = support_name_rgb.replace('.png','_rgb.png')########################3
        support_name_th = support_name.replace('.png','_th.png')
        support_name_d = support_name.replace('.png', '_d.png')
        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(random.uniform(1, 1.5)*input_size)
        #scale_transform_mask = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        #scale_transform_rgb = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        #scale_transform_th = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        #scale_transform_d = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = random.random()

        # image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        # image_th = Image.fromarray(cv2.cvtColor(image_th,cv2.COLOR_BGR2RGB))
        image_th = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        image_d = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_d)).convert('RGB')

        support_th = self.normalize(
            self.ToTensor(
                    self.transs(image_th)))

        support_d = self.normalize(
            self.ToTensor(
                    self.transs(image_d)))

        support_rgb = self.normalize(
            self.ToTensor(
                    self.transs(Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', support_name_rgb)))))

        support_mask = self.ToTensor(Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           support_name )))

        '''margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)
        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_d = support_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]'''

        # random scale and crop for query
        scaled_size = input_size  # random.randint(323, 350)
        scale_transform_d = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_th = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_mask = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0  # random.random()'
        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png','_rgb.png')######
        query_name_th = query_name.replace('.png','_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        # image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        # image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))
        image_thq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        image_dq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_d)).convert('RGB')

        query_th = self.normalize(
            self.ToTensor(
                    self.transq(image_thq)))

        query_d = self.normalize(
            self.ToTensor(
                    self.transq(image_dq)))

        query_rgb = self.normalize(
            self.ToTensor(
                    self.transq(Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', query_name_rgb )))))

        query_mask = self.ToTensor(
                self.transq(Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           query_name))))

        '''margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]'''

        return query_rgb, query_th, query_d, query_mask.long(), support_rgb, support_th, support_d, support_mask.long(), sample_class-1

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return len(self.new_exist_class_list)


class Tri_Dataset(object):

    def __init__(self, data_dir, fold, input_size=[400, 400], 
                 normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1],
                 normalize_mean_d=[0, 0, 0], normalize_std_d=[1, 1, 1],
                 normalize_mean_th=[0, 0, 0], normalize_std_th=[1, 1, 1],
                 prob=0.7):
        # -------------------load data list,[class,video_name]-------------------
        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)
        self.initiaize_transformation(normalize_mean, normalize_std, normalize_mean_d, normalize_std_d, normalize_mean_th, normalize_std_th, input_size)
        self.data_enhancement(tuple(input_size))
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.prob = prob  # probability of sampling history masks=0
        self.split = fold

        if self.split == 3:
            self.sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        elif self.split == 2:
            self.sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
        elif self.split == 1:
            self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
        elif self.split == 0:
            self.sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []
        fold_list = [0, 1, 2, 3]
        fold_list.remove(fold)
        for fold in fold_list:
            f = open(os.path.join(self.data_dir, 'Binary_map','split%1d.txt'%fold))
            while True:
                item = f.readline()
                if item == '':
                    break
                item2 = item.split("_")
                img_name = item2[0]
                cat = int(item2[1])
                new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def data_enhancement(self,img_size):
        self.transs = transforms.Compose([
                            transforms.RandomRotation((-10, 10)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(img_size),
                            ])
        self.transq = transforms.Compose([
                            transforms.RandomRotation((-10, 10)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(img_size,ratio=(0.9,1.1)),
                            ])

    def initiaize_transformation(self, normalize_mean, normalize_std, normalize_mean_d, normalize_std_d, normalize_mean_th, normalize_std_th, input_size):
        self.ToTensor = transforms.ToTensor()
        self.resize = transforms.Resize(input_size)
        self.normalize = transforms.Normalize(normalize_mean, normalize_std)
        self.normalize_d = torchvision.transforms.Normalize(normalize_mean_d, normalize_std_d)
        self.normalize_th = torchvision.transforms.Normalize(normalize_mean_th, normalize_std_th)

    def get_binary_pair_list(self):  # a list store all img name that contain that class
        binary_pair_list = {}
        for Class in range(1, 21):
            binary_pair_list[Class] = self.read_txt(
                os.path.join(self.data_dir, 'Binary_map', '%d.txt' % Class))
        return binary_pair_list

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        return out_list

    def __getitem__(self, index):

        # give an query index,sample a target class first
        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]  # random sample a class in this img

        support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
        while True:  # random sample a support data
            support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                break

        # print (query_name,support_name)
        support_name_rgb= support_name
        support_name_rgb = support_name_rgb.replace('.png','_rgb.png')########################3
        support_name_th = support_name.replace('.png','_th.png')
        support_name_d = support_name.replace('.png', '_d.png')
        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(random.uniform(1, 1.5)*input_size)
        scale_transform_mask = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_d = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = random.random()

        # image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        # image_th = Image.fromarray(cv2.cvtColor(image_th,cv2.COLOR_BGR2RGB))
        image_th = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        image_d = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_d)).convert('RGB')

        support_th = self.normalize_th(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag,image_th))))

        support_d = self.normalize_d(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag,image_d))))

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', support_name_rgb))))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           support_name )))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)
        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_d = support_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        # random scale and crop for query
        scaled_size = input_size  # random.randint(323, 350)
        scale_transform_d = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_th = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_mask = transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0  # random.random()'
        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png','_rgb.png')######
        query_name_th = query_name.replace('.png','_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        # image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        # image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))
        image_thq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        image_dq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_d)).convert('RGB')

        query_th = self.normalize_th(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag,image_thq))))

        query_d = self.normalize_d(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag,image_dq))))

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', query_name_rgb ))))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           query_name)))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        return query_rgb, query_th, query_d, query_mask.long(), support_rgb, support_th, support_d, support_mask.long(), sample_class-1

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return len(self.new_exist_class_list)



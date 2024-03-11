import random
import os
import torchvision
import torch
import cv2
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as F_tensor
import numpy as np
from torch.utils.data import DataLoader
import time

class Dataset(object):

    def __init__(self, data_dir, fold, input_size=[400, 400], normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1],inference=False):
        self.inference = inference

        self.data_dir = data_dir
        self.input_size = input_size

        # random sample 1000 pairs
        self.chosen_data_list_1 = self.get_new_exist_class_dict(fold=fold)
        chosen_data_list_2 = self.chosen_data_list_1[:]
        chosen_data_list_3 = self.chosen_data_list_1[:]
        chosen_data_list_4= self.chosen_data_list_1[:]
        chosen_data_list_5 = self.chosen_data_list_1[:]
        chosen_data_list_6 = self.chosen_data_list_1[:]
        random.shuffle(chosen_data_list_2)
        random.shuffle(chosen_data_list_3)
        random.shuffle(chosen_data_list_4)
        random.shuffle(chosen_data_list_5)
        random.shuffle(chosen_data_list_6)

        self.chosen_data_list=self.chosen_data_list_1+chosen_data_list_2+chosen_data_list_3+chosen_data_list_4+chosen_data_list_5+chosen_data_list_6
        self.chosen_data_list=self.chosen_data_list[:1000]

        self.split = fold
        self.binary_pair_list = self.get_binary_pair_list()#a dict of each class, which contains all imgs that include this class
        self.query_class_support_list=[None] * 1000
        for index in range (1000):
            query_name=self.chosen_data_list[index][0]
            sample_class=self.chosen_data_list[index][1]
            support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
            while True:  # random sample a support data
                support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
                if support_name != query_name:
                    break
            self.query_class_support_list[index] = [query_name,sample_class,support_name]

        if self.split == 3:
            self.sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
        elif self.split == 2:
            self.sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
        elif self.split == 1:
            self.sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
        elif self.split == 0:
            self.sub_val_list = list(range(1, 6))  # [1,2,3,4,5]
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        pass

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []
        f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % (fold)))
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
        self.ToTensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)

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
        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]  # random sample a class in this img
        support_name=self.query_class_support_list[index][2]

        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(random.uniform(1,1.5)*input_size)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_d = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = random.random()
        support_name_rgb = support_name
        support_name_rgb = support_name_rgb.replace('.png', '_rgb.png')
        support_name_th = support_name.replace('.png', '_th.png')
        support_name_d = support_name.replace('.png', '_d.png')

        # image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        # image_th = Image.fromarray(cv2.cvtColor(image_th,cv2.COLOR_BGR2RGB))
        image_th = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        image_d = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_d)).convert('RGB')

        support_th = self.normalize(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, image_th))))

        support_d = self.normalize(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag, image_d))))

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', support_name_rgb))))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class), support_name )))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_d = support_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        # random scale and crop for query
        scaled_size = 400

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_d = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0  # random.random()
        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png','_rgb.png')
        query_name_th = query_name.replace('.png', '_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        # image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        # image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))
        image_thq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        image_dq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_d)).convert('RGB')

        query_th = self.normalize(
            self.ToTensor(
                scale_transform_th(image_thq)))

        query_d = self.normalize(
            self.ToTensor(
                scale_transform_d((image_dq))))

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', query_name_rgb)))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                          Image.open(
                              os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           query_name))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        if(self.inference):
            return query_rgb, query_th, query_d, query_mask.long(), support_rgb,support_th, support_d, support_mask.long(), sample_class-1, query_name_rgb
        else:
            return query_rgb, query_th, query_d, query_mask.long(), support_rgb,support_th, support_d, support_mask.long(), sample_class-1

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return 1000



class Tri_Dataset(object):

    def __init__(self, data_dir, fold, input_size=[400, 400], 
                 normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1],
                 normalize_mean_d=[0, 0, 0], normalize_std_d=[1, 1, 1],
                 normalize_mean_th=[0, 0, 0], normalize_std_th=[1, 1, 1],
                 mode = 'train'):
        self.mode = mode

        self.data_dir = data_dir
        self.input_size = input_size

        # random sample 1000 pairs
        self.chosen_data_list_1 = self.get_new_exist_class_dict(fold=fold)
        chosen_data_list_2 = self.chosen_data_list_1[:]
        chosen_data_list_3 = self.chosen_data_list_1[:]
        chosen_data_list_4= self.chosen_data_list_1[:]
        chosen_data_list_5 = self.chosen_data_list_1[:]
        chosen_data_list_6 = self.chosen_data_list_1[:]
        random.shuffle(chosen_data_list_2)
        random.shuffle(chosen_data_list_3)
        random.shuffle(chosen_data_list_4)
        random.shuffle(chosen_data_list_5)
        random.shuffle(chosen_data_list_6)

        self.chosen_data_list=self.chosen_data_list_1+chosen_data_list_2+chosen_data_list_3+chosen_data_list_4+chosen_data_list_5+chosen_data_list_6
        self.chosen_data_list=self.chosen_data_list[:1000]

        self.split = fold
        self.binary_pair_list = self.get_binary_pair_list()#a dict of each class, which contains all imgs that include this class
        self.query_class_support_list=[None] * 1000
        for index in range (1000):
            query_name=self.chosen_data_list[index][0]
            sample_class=self.chosen_data_list[index][1]
            support_img_list = self.binary_pair_list[sample_class]  # all img that contain the sample_class
            while True:  # random sample a support data
                support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
                if support_name != query_name:
                    break
            self.query_class_support_list[index] = [query_name,sample_class,support_name]

        if self.split == 3:
            self.sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
        elif self.split == 2:
            self.sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
        elif self.split == 1:
            self.sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
        elif self.split == 0:
            self.sub_val_list = list(range(1, 6))  # [1,2,3,4,5]
        self.initiaize_transformation(normalize_mean, normalize_std, normalize_mean_d, normalize_std_d, normalize_mean_th, normalize_std_th)
        

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []
        f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % (fold)))
        while True:
            item = f.readline()
            if item == '':
                break
            item2 = item.split("_")
            img_name = item2[0]
            cat = int(item2[1])
            new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def initiaize_transformation(self, normalize_mean, normalize_std, normalize_mean_d, normalize_std_d, normalize_mean_th, normalize_std_th):
        self.ToTensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(normalize_mean, normalize_std)
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
        query_name = self.query_class_support_list[index][0]
        sample_class = self.query_class_support_list[index][1]  # random sample a class in this img
        support_name=self.query_class_support_list[index][2]

        input_size = self.input_size[0]
        # random scale and crop for support
        scaled_size = int(random.uniform(1,1.5)*input_size)
        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_d = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = random.random()
        support_name_rgb = support_name
        support_name_rgb = support_name_rgb.replace('.png', '_rgb.png')
        support_name_th = support_name.replace('.png', '_th.png')
        support_name_d = support_name.replace('.png', '_d.png')

        # image_th = cv2.imread(os.path.join(self.data_dir, 'seperated_images', support_name_th))
        # image_th = Image.fromarray(cv2.cvtColor(image_th,cv2.COLOR_BGR2RGB))
        image_path_ths = os.path.join(self.data_dir, 'seperated_images', support_name_th)
        image_path_ds = os.path.join(self.data_dir, 'seperated_images', support_name_d)
        image_path_vs = os.path.join(self.data_dir, 'seperated_images', support_name_rgb)
        image_path_masks = os.path.join(self.data_dir, 'Binary_map', str(sample_class),support_name)

        support_th = self.normalize_th(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, Image.open(image_path_ths)))))

        support_d = self.normalize_d(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag, Image.open(image_path_ds).convert('RGB')))))

        support_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag,
                              Image.open(image_path_vs)))))

        support_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag,
                          Image.open(image_path_masks))))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_mask = support_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        support_d = support_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        # random scale and crop for query
        scaled_size = 400

        scale_transform_mask = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.NEAREST)
        scale_transform_rgb = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_th = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        scale_transform_d = torchvision.transforms.Resize([scaled_size, scaled_size], interpolation=Image.BILINEAR)
        flip_flag = 0  # random.random()
        query_name_rgb = query_name
        query_name_rgb = query_name_rgb.replace('.png','_rgb.png')
        query_name_th = query_name.replace('.png', '_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        # image_thq = cv2.imread(os.path.join(self.data_dir, 'seperated_images', query_name_th))
        # image_thq = Image.fromarray(cv2.cvtColor(image_thq, cv2.COLOR_BGR2RGB))
        image_path_thq = os.path.join(self.data_dir, 'seperated_images', query_name_th)
        image_path_dq = os.path.join(self.data_dir, 'seperated_images', query_name_d)
        image_path_vq = os.path.join(self.data_dir, 'seperated_images', query_name_rgb)
        image_path_maskq = os.path.join(self.data_dir, 'Binary_map', str(sample_class),
                                           query_name)


        query_th = self.normalize_th(
            self.ToTensor(
                scale_transform_th(Image.open(image_path_thq))))

        query_d = self.normalize_d(
            self.ToTensor(
                scale_transform_d((Image.open(image_path_dq).convert('RGB')))))

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                              Image.open(image_path_vq))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                          Image.open(image_path_maskq)))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        if(self.mode == 'vis'):
            ori = self.ToTensor(
                scale_transform_rgb(
                              Image.open(
                                  os.path.join(self.data_dir, 'seperated_images', query_name_rgb))))
            return query_rgb, query_th, query_d, query_mask.long(), support_rgb,support_th, support_d, support_mask.long(), sample_class-1, query_name_rgb, ori
        elif(self.mode == 'test'):
            return query_rgb, query_th, query_d, query_mask.long(), support_rgb,support_th, support_d, support_mask.long(), sample_class-1, image_path_maskq,image_path_vq,image_path_dq,image_path_thq, image_path_masks,image_path_vs,image_path_ds,image_path_ths
        else:
            return query_rgb, query_th, query_d, query_mask.long(), support_rgb,support_th, support_d, support_mask.long(), sample_class-1

    def flip(self, flag, img):
        if flag > 0.5:
            return F.hflip(img)
        else:
            return img

    def __len__(self):
        return 1000

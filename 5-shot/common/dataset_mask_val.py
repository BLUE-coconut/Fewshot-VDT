import random
import os
from PIL import Image
import numpy as np
import time

import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.transforms.py_transforms as PP

random.seed(0)


class Dataset(object):

    def __init__(self, data_dir, fold, shot, input_size=[400, 400], normalize_mean=[0, 0, 0],
                 normalize_std=[1, 1, 1]):

        self.data_dir = data_dir
        self.shot = shot
        self.input_size = input_size

        self.chosen_data_list_1 = self.get_new_exist_class_dict(fold=fold)
        chosen_data_list_2 = self.chosen_data_list_1[:]
        chosen_data_list_3 = self.chosen_data_list_1[:]
        chosen_data_list_4 = self.chosen_data_list_1[:]
        chosen_data_list_5 = self.chosen_data_list_1[:]
        chosen_data_list_6 = self.chosen_data_list_1[:]
        random.shuffle(chosen_data_list_2)
        random.shuffle(chosen_data_list_3)
        random.shuffle(chosen_data_list_4)
        random.shuffle(chosen_data_list_5)
        random.shuffle(chosen_data_list_6)

        self.chosen_data_list = self.chosen_data_list_1 + chosen_data_list_2 + chosen_data_list_3 + \
                                chosen_data_list_4 + chosen_data_list_5 + chosen_data_list_6
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)

        self.sub_list = list(range(1, 21))

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []
        f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % fold))
        while True:
            item = f.readline()
            if item == '':
                break
            item2 = item.split("_")
            img_name = item2[0]
            cat = int(item2[1])
            new_exist_class_list.append([img_name, cat])
        f.close()
        return new_exist_class_list

    def initiaize_transformation(self, normalize_mean, normalize_std, input_size):
        self.ToTensor = P.ToTensor()
        self.resize_bilinear = P.Resize(input_size, interpolation=P.Inter.BILINEAR)
        self.resize_nearest = P.Resize(input_size, interpolation=P.Inter.NEAREST)

        self.norm_V = P.Normalize(normalize_mean, normalize_std)
        self.norm_T = P.Normalize(normalize_mean, normalize_std)
        self.norm_D = P.Normalize(normalize_mean, normalize_std)

    def read_txt(self, dir):
        f = open(dir)
        out_list = []
        line = f.readline()
        while line:
            out_list.append(line.split()[0])
            line = f.readline()
        f.close()
        return out_list

    def __getitem__(self, index):

        query_name = self.chosen_data_list[index][0]
        sample_class = self.chosen_data_list[index][1]

        support_img_list = self.read_txt(
            os.path.join(self.data_dir, 'Binary_map', '%d.txt' % sample_class))

        support_names = []
        while True:
            support_name = support_img_list[random.randint(0, len(support_img_list) - 1)]
            if support_name != query_name:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break

        support_rgbs = []
        support_masks = []
        support_ths = []
        support_ds = []

        scale_transform_rgb = self.resize_bilinear
        scale_transform_th = self.resize_bilinear
        scale_transform_d = self.resize_bilinear
        scale_transform_mask = self.resize_nearest

        for support_name in support_names:
            support_name_rgb = support_name.replace('.png', '_rgb.png')
            support_name_th = support_name.replace('.png', '_th.png')
            support_name_d = support_name.replace('.png', '_d.png')

            image_path_th = os.path.join(self.data_dir, 'seperated_images', support_name_th)
            image_path_d = os.path.join(self.data_dir, 'seperated_images', support_name_d)
            image_path_rgb = os.path.join(self.data_dir, 'seperated_images', support_name_rgb)
            image_path_mask = os.path.join(self.data_dir, 'Binary_map', str(sample_class), support_name)

            ori_support_th = Image.open(image_path_th)
            ori_support_d = Image.open(image_path_d)
            ori_support_rgb = Image.open(image_path_rgb)
            ori_support_mask = Image.open(image_path_mask).convert('L')

            support_th = self.norm_T(scale_transform_th(ori_support_th))
            support_d = self.norm_D(scale_transform_d(ori_support_d.convert('RGB')))
            support_rgb = self.norm_V(scale_transform_rgb(ori_support_rgb))
            support_mask = self.ToTensor(scale_transform_mask(ori_support_mask))

            support_rgbs.append(support_rgb)
            support_ths.append(support_th)
            support_ds.append(support_d)
            support_masks.append(support_mask[:1])

        support_rgb = np.stack(support_rgbs)
        support_d = np.stack(support_ds)
        support_th = np.stack(support_ths)
        support_mask = np.stack(support_masks)

        query_name_rgb = query_name.replace('.png', '_rgb.png')
        query_name_th = query_name.replace('.png', '_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        image_path_thq = os.path.join(self.data_dir, 'seperated_images', query_name_th)
        image_path_dq = os.path.join(self.data_dir, 'seperated_images', query_name_d)
        image_path_vq = os.path.join(self.data_dir, 'seperated_images', query_name_rgb)
        image_path_maskq = os.path.join(self.data_dir, 'Binary_map', str(sample_class), query_name)

        ori_query_th = Image.open(image_path_thq)
        ori_query_d = Image.open(image_path_dq)
        ori_query_rgb = Image.open(image_path_vq)
        ori_query_mask = Image.open(image_path_maskq).convert('L')

        ori_query_imsize = np.array(ori_query_rgb.size[::-1])

        query_th = self.norm_T(scale_transform_th(ori_query_th))
        query_d = self.norm_D(scale_transform_d(ori_query_d.convert('RGB')))
        query_rgb = self.norm_V(scale_transform_rgb(ori_query_rgb))
        query_mask = self.ToTensor(scale_transform_mask(ori_query_mask))[:1]

        return (query_rgb, query_th, query_d, query_mask.astype(np.int64),
                support_rgb, support_th, support_d, support_mask.astype(np.int64),
                np.array(sample_class - 1).astype(np.int64),
                ori_query_imsize,
                query_name.split('.')[0]
                )

    def __len__(self):
        return len(self.chosen_data_list)
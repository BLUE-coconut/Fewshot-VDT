import random
import os
from PIL import Image
import numpy as np

import mindspore.dataset.vision.py_transforms as P

class Dataset(object):

    def __init__(self, data_dir, fold, shot, input_size=[400, 400],
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225], prob=0.7):

        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)
        self.initiaize_transformation(normalize_mean, normalize_std, input_size)
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.prob = prob
        self.split = fold
        self.shot = shot
        print(self.split)

        if self.split == 3:
            self.sub_list = list(range(1, 16))
        elif self.split == 2:
            self.sub_list = list(range(1, 11)) + list(range(16, 21))
        elif self.split == 1:
            self.sub_list = list(range(1, 6)) + list(range(11, 21))
        elif self.split == 0:
            self.sub_list = list(range(6, 21))

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []
        fold_list = [0, 1, 2, 3]
        fold_list.remove(fold)
        for fold_idx in fold_list:
            f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % fold_idx))
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
        self.ToTensor = P.ToTensor()
        self.resize = P.Resize(input_size, interpolation=P.Inter.BILINEAR)
        self.normalize = P.Normalize(normalize_mean, normalize_std)

    def get_binary_pair_list(self):
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

    def flip(self, flag, img):
        from PIL import Image
        if flag > 0.5:
            if img.mode in ['L', 'RGB']:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img
        else:
            return img

    def __getitem__(self, index):

        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]

        support_img_list = self.binary_pair_list[sample_class]
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
        input_size = self.input_size[0]

        for support_name in support_names:
            support_name_rgb = support_name.replace('.png', '_rgb.png')
            support_name_th = support_name.replace('.png', '_th.png')
            support_name_d = support_name.replace('.png', '_d.png')

            # random scale and crop for support
            scaled_size = int(random.uniform(1, 1.5) * input_size)
            scale_transform_mask = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.NEAREST)
            scale_transform_rgb = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
            scale_transform_th = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
            scale_transform_d = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
            flip_flag = random.random()

            image_th = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_th)).convert('RGB')
            image_d = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_d)).convert('RGB')
            image_rgb = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_rgb)).convert('RGB')
            support_mask_img = Image.open(
                os.path.join(self.data_dir, 'Binary_map', str(sample_class), support_name)).convert('L')

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
                        self.flip(flip_flag, image_rgb))))

            support_mask = self.ToTensor(
                scale_transform_mask(
                    self.flip(flip_flag, support_mask_img)))

            margin_h = random.randint(0, scaled_size - input_size)
            margin_w = random.randint(0, scaled_size - input_size)

            support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_mask = support_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_d = support_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

            support_rgbs.append(support_rgb)
            support_ths.append(support_th)
            support_ds.append(support_d)
            support_masks.append(support_mask)

        support_rgb = np.stack(support_rgbs)
        support_d = np.stack(support_ds)
        support_th = np.stack(support_ths)
        support_mask = np.stack(support_masks)

        scaled_size = input_size
        scale_transform_d = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
        scale_transform_th = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
        scale_transform_mask = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.NEAREST)
        scale_transform_rgb = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
        flip_flag = 0
        query_name_rgb = query_name.replace('.png', '_rgb.png')
        query_name_th = query_name.replace('.png', '_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        image_thq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_th)).convert('RGB')
        image_dq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_d)).convert('RGB')
        image_rgbq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_rgb)).convert('RGB')
        query_mask_img = Image.open(os.path.join(self.data_dir, 'Binary_map', str(sample_class), query_name)).convert(
            'L')

        query_th = self.normalize(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, image_thq))))

        query_d = self.normalize(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag, image_dq))))

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag, image_rgbq))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag, query_mask_img)))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        return (query_rgb, query_th, query_d, query_mask.astype(np.int64),
                support_rgb, support_th, support_d, support_mask.astype(np.int64),
                np.array(sample_class - 1).astype(np.int64))

    def __len__(self):
        return len(self.new_exist_class_list)

class Tri_Dataset(object):

    def __init__(self, data_dir, fold, shot, input_size=[400, 400],
                 normalize_mean=[0, 0, 0], normalize_std=[1, 1, 1],
                 normalize_mean_d=[0, 0, 0], normalize_std_d=[1, 1, 1],
                 normalize_mean_th=[0, 0, 0], normalize_std_th=[1, 1, 1],
                 prob=0.7):
        self.data_dir = data_dir
        self.new_exist_class_list = self.get_new_exist_class_dict(fold=fold)
        self.initiaize_transformation(normalize_mean, normalize_std, normalize_mean_d, normalize_std_d,
                                      normalize_mean_th, normalize_std_th, input_size)
        self.binary_pair_list = self.get_binary_pair_list()
        self.input_size = input_size
        self.prob = prob
        self.split = fold
        self.shot = shot

        if self.split == 3:
            self.sub_list = list(range(1, 16))
        elif self.split == 2:
            self.sub_list = list(range(1, 11)) + list(range(16, 21))
        elif self.split == 1:
            self.sub_list = list(range(1, 6)) + list(range(11, 21))
        elif self.split == 0:
            self.sub_list = list(range(6, 21))

    def get_new_exist_class_dict(self, fold):
        new_exist_class_list = []
        fold_list = [0, 1, 2, 3]
        fold_list.remove(fold)
        for fold_idx in fold_list:
            f = open(os.path.join(self.data_dir, 'Binary_map', 'split%1d.txt' % fold_idx))
            while True:
                item = f.readline()
                if item == '':
                    break
                item2 = item.split("_")
                img_name = item2[0]
                cat = int(item2[1])
                new_exist_class_list.append([img_name, cat])
        return new_exist_class_list

    def initiaize_transformation(self, normalize_mean, normalize_std, normalize_mean_d, normalize_std_d,
                                 normalize_mean_th, normalize_std_th, input_size):
        self.ToTensor = P.ToTensor()
        self.resize = P.Resize(input_size, interpolation=P.Inter.BILINEAR)
        self.normalize = P.Normalize(normalize_mean, normalize_std)
        self.normalize_d = P.Normalize(normalize_mean_d, normalize_std_d)
        self.normalize_th = P.Normalize(normalize_mean_th, normalize_std_th)

    def get_binary_pair_list(self):
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

    def flip(self, flag, img):
        from PIL import Image
        if flag > 0.5:
            if img.mode in ['L', 'RGB']:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img
        else:
            return img

    def __getitem__(self, index):
        query_name = self.new_exist_class_list[index][0]
        sample_class = self.new_exist_class_list[index][1]

        support_img_list = self.binary_pair_list[sample_class]
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
        input_size = self.input_size[0]

        for support_name in support_names:
            support_name_rgb = support_name.replace('.png', '_rgb.png')
            support_name_th = support_name.replace('.png', '_th.png')
            support_name_d = support_name.replace('.png', '_d.png')

            scaled_size = int(random.uniform(1, 1.5) * input_size)
            scale_transform_mask = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.NEAREST)
            scale_transform_rgb = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
            scale_transform_th = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
            scale_transform_d = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
            flip_flag = random.random()

            image_th = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_th)).convert('RGB')
            image_d = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_d)).convert('RGB')
            image_rgb = Image.open(os.path.join(self.data_dir, 'seperated_images', support_name_rgb)).convert('RGB')
            support_mask_img = Image.open(
                os.path.join(self.data_dir, 'Binary_map', str(sample_class), support_name)).convert('L')

            support_th = self.normalize_th(
                self.ToTensor(
                    scale_transform_th(
                        self.flip(flip_flag, image_th))))

            support_d = self.normalize_d(
                self.ToTensor(
                    scale_transform_d(
                        self.flip(flip_flag, image_d))))

            support_rgb = self.normalize(
                self.ToTensor(
                    scale_transform_rgb(
                        self.flip(flip_flag, image_rgb))))

            support_mask = self.ToTensor(
                scale_transform_mask(
                    self.flip(flip_flag, support_mask_img)))

            margin_h = random.randint(0, scaled_size - input_size)
            margin_w = random.randint(0, scaled_size - input_size)

            support_rgb = support_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_mask = support_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_th = support_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
            support_d = support_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

            support_rgbs.append(support_rgb)
            support_ths.append(support_th)
            support_ds.append(support_d)
            support_masks.append(support_mask)

        support_rgb = np.stack(support_rgbs)
        support_d = np.stack(support_ds)
        support_th = np.stack(support_ths)
        support_mask = np.stack(support_masks)

        scaled_size = input_size
        scale_transform_d = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
        scale_transform_th = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
        scale_transform_mask = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.NEAREST)
        scale_transform_rgb = P.Resize([scaled_size, scaled_size], interpolation=P.Inter.BILINEAR)
        flip_flag = 0
        query_name_rgb = query_name.replace('.png', '_rgb.png')
        query_name_th = query_name.replace('.png', '_th.png')
        query_name_d = query_name.replace('.png', '_d.png')

        image_thq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_th)).convert('RGB')
        image_dq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_d)).convert('RGB')
        image_rgbq = Image.open(os.path.join(self.data_dir, 'seperated_images', query_name_rgb)).convert('RGB')
        query_mask_img = Image.open(os.path.join(self.data_dir, 'Binary_map', str(sample_class), query_name)).convert(
            'L')

        query_th = self.normalize_th(
            self.ToTensor(
                scale_transform_th(
                    self.flip(flip_flag, image_thq))))

        query_d = self.normalize_d(
            self.ToTensor(
                scale_transform_d(
                    self.flip(flip_flag, image_dq))))

        query_rgb = self.normalize(
            self.ToTensor(
                scale_transform_rgb(
                    self.flip(flip_flag, image_rgbq))))

        query_mask = self.ToTensor(
            scale_transform_mask(
                self.flip(flip_flag, query_mask_img)))

        margin_h = random.randint(0, scaled_size - input_size)
        margin_w = random.randint(0, scaled_size - input_size)

        query_rgb = query_rgb[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_mask = query_mask[:1, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_th = query_th[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]
        query_d = query_d[:, margin_h:margin_h + input_size, margin_w:margin_w + input_size]

        return (query_rgb, query_th, query_d, query_mask.astype(np.int64),
                support_rgb, support_th, support_d, support_mask.astype(np.int64),
                np.array(sample_class - 1).astype(np.int64))

    def __len__(self):
        return len(self.new_exist_class_list)
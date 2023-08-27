import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFilter 
import PIL
import os, cv2
import numpy as np
import random
import copy
import json

cudnn.deterministic = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

def random_translate(image, target):
    image_width, image_height = image.size

    xmin, ymin = np.min(target.reshape(-1, 2), axis=0)
    xmax, ymax = np.max(target.reshape(-1, 2), axis=0)

    a = 1
    b = 0
    # refer to https://pillow.readthedocs.io/en/stable/_modules/PIL/ImageTransform.html
    # left margin, positive
    c_max = xmin * image_width
    # right margin, negative
    c_min = -(1-xmax) * image_width
    c = random.uniform(c_min, c_max)
    d = 0
    e = 1
    # top margin, positive
    f_max = ymin * image_height
    # bottom margin, negative
    f_min = -(1-ymax) * image_height
    f = random.uniform(f_min, f_max)

    image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
    target_translate = target.copy()
    target_translate = target_translate.reshape(-1, 2)
    target_translate[:, 0] -= 1.*c/image_width
    target_translate[:, 1] -= 1.*f/image_height
    target_translate = target_translate.flatten()
    return image, target_translate

def random_rotate(image, target, angle_max):
    #######################################
    image_width, image_height = image.size
    image = image.resize((image_height, image_height), PIL.Image.BICUBIC)
    #######################################
    center_x = 0.5
    center_y = 0.5
    landmark_num= int(len(target) / 2)
    target_center = np.array(target) - np.array([center_x, center_y]*landmark_num)
    target_center = target_center.reshape(landmark_num, 2)
    theta_max = np.radians(angle_max)
    theta = random.uniform(-theta_max, theta_max)
    angle = np.degrees(theta)
    image = image.rotate(angle)

    c, s = np.cos(theta), np.sin(theta)
    rot = np.array(((c,-s), (s, c)))
    target_center_rot = np.matmul(target_center, rot)
    target_rot = target_center_rot.reshape(landmark_num*2) + np.array([center_x, center_y]*landmark_num)
    #######################################
    image = image.resize((image_width, image_height), PIL.Image.BICUBIC)
    #######################################
    return image, target_rot

def random_scale(image, target, scale_min=0.8, scale_max=1.1, aspect_ratio_min=0.8, aspect_ratio_max=1.2):
    image_np = np.array(image).astype(np.uint8)
    image_np = image_np[:,:,::-1]
    image_height, image_width, _ = image_np.shape
    scale = random.uniform(scale_min, scale_max)
    # scale_w / scale_h
    aspect_ratio = random.uniform(aspect_ratio_min, aspect_ratio_max)
    if scale < 1: # i.e.: padding margins
        pad_h = int(image_height * (1-scale) / 2)
        pad_w = int(image_width * (1-scale) / 2 * aspect_ratio)

        padding_h = np.zeros((pad_h, image_width, 3))
        padding_w = np.zeros((image_height+2*pad_h, pad_w, 3))

        image_np = np.concatenate([padding_h, image_np, padding_h], axis=0)
        image_np = np.concatenate([padding_w, image_np, padding_w], axis=1)

        target_scale = target.copy()
        target_scale = target_scale.reshape(-1, 2)
        target_scale *= np.array([[image_width, image_height]])
        target_scale += np.array([[pad_w, pad_h]])

        image_height_new, image_width_new, _ = image_np.shape
        target_scale /= np.array([[image_width_new, image_height_new]])

        image_np = cv2.resize(image_np, (image_width, image_height), interpolation=cv2.INTER_AREA)
        image_pil = Image.fromarray(image_np[:,:,::-1].astype('uint8'), 'RGB')
        return image_pil, target_scale.flatten()
    elif scale > 1:
        crop_h = image_height * (scale-1) / 2
        crop_w = image_width * (scale-1) / 2 * aspect_ratio

        xmin, ymin = np.min(target.reshape(-1, 2), axis=0)
        xmax, ymax = np.max(target.reshape(-1, 2), axis=0)

        crop_h = min(crop_h, ymin*image_height-5)
        crop_h = min(crop_h, (1-ymax)*image_height-5)
        crop_w = min(crop_w, xmin*image_width-5)
        crop_w = min(crop_w, (1-xmax)*image_width-5)

        crop_h = max(crop_h, 0)
        crop_w = max(crop_w, 0)

        crop_h = int(crop_h)
        crop_w = int(crop_w)

        image_np = image_np[crop_h:(image_height-crop_h)+1, crop_w:(image_width-crop_w)+1, :]

        target_scale = target.copy()
        target_scale = target_scale.reshape(-1, 2)
        target_scale *= np.array([[image_width, image_height]])
        target_scale -= np.array([[crop_w, crop_h]])

        image_height_new, image_width_new, _ = image_np.shape
        target_scale /= np.array([[image_width_new, image_height_new]])

        image_np = cv2.resize(image_np, (image_width, image_height), interpolation=cv2.INTER_AREA)
        image_pil = Image.fromarray(image_np[:,:,::-1].astype('uint8'), 'RGB')
        return image_pil, target_scale.flatten()
    else:
        return image, target

def random_blur(image):
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.random()*2))
    return image

def random_occlusion(image):
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_np = image_np[:,:,::-1]
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height*0.3*random.random())
        occ_width = int(image_width*0.3*random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, :] = 0
        image_pil = Image.fromarray(image_np[:,:,::-1].astype('uint8'), 'RGB')
        return image_pil
    else:
        return image

def random_flip(image, target, points_flip):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        target = np.array(target).reshape(-1, 2)
        target = target[points_flip, :]
        target[:,0] = 1-target[:,0]
        target = target.flatten()
        return image, target
    else:
        return image, target

def gen_target_map(target, target_map, target_local_x, target_local_y, sigma):
    map_channel, map_height, map_width = target_map.shape
    target = target.reshape(-1, 2)
    assert map_channel == target.shape[0]
    tmp_size = sigma * 3
    for i in range(map_channel):
        #################################
        # heatmap
        mu_x = int(target[i][0] * map_width - 0.5)
        mu_y = int(target[i][1] * map_height - 0.5)
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, map_width-1)
        mu_y = min(mu_y, map_height-1)
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size), int(mu_y + tmp_size)]
        ul[0] = max(0, ul[0])
        ul[1] = max(0, ul[1])
        br[0] = min(br[0], map_width-1)
        br[1] = min(br[1], map_height-1)
        margin_left = int(min(mu_x, tmp_size))
        margin_right = int(min(map_width-1-mu_x, tmp_size))
        margin_top = int(min(mu_y, tmp_size))
        margin_bottom = int(min(map_height-1-mu_y, tmp_size))
        assert margin_right >= -margin_left
        assert margin_bottom >= -margin_top

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = int(size // 2)
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        target_map[i, (mu_y-margin_top):(mu_y+margin_bottom+1), (mu_x-margin_left):(mu_x+margin_right+1)] = g[(y0-margin_top):(y0+margin_bottom+1), (x0-margin_left):(x0+margin_right+1)]
        #################################
        for j in range(mu_y-margin_top, mu_y+margin_bottom+1):
            for k in range(mu_x-margin_left, mu_x+margin_right+1):
                shift_x = target[i][0]*map_width - k
                shift_y = target[i][1]*map_height - j
                target_local_x[i, j, k] = shift_x
                target_local_y[i, j, k] = shift_y
        #################################
    return target_map, target_local_x, target_local_y

class ImageFolder_tf(data.Dataset):
    def __init__(self, cfg, phase, labels, map_size, transform=None):
        self.cfg = cfg
        self.phase = phase
        self.labels = labels
        self.map_size = map_size
        self.transform = transform
        self.num_lms = cfg.num_lms
        self.input_size = cfg.input_size
        self.gt_sigma = cfg.gt_sigma
        if self.phase == 'val' and self.cfg.task == 'head':
            with open('data/HeadNew/img2dist.json', 'r') as f:
                self.img2dist = json.load(f)
            with open('data/HeadNew/img2size.json', 'r') as f:
                self.img2size = json.load(f)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path, target, mask, domain = self.labels[index]

        img = Image.open(img_path).convert('RGB')
        if self.phase == 'train':
            img, target = random_scale(img, target)
            img, target = random_translate(img, target)
            img = random_occlusion(img)
            img, target = random_rotate(img, target, 50)
            img = random_blur(img)

            target_map = np.zeros((self.num_lms, self.map_size[2], self.map_size[3]))
            target_local_x = np.zeros((self.num_lms, self.map_size[2], self.map_size[3]))
            target_local_y = np.zeros((self.num_lms, self.map_size[2], self.map_size[3]))

            target_map, target_local_x, target_local_y = gen_target_map(target, target_map, target_local_x, target_local_y, self.gt_sigma)

            mask_map = np.ones_like(target_map)
            mask_xy = np.ones((self.num_lms, 21))
            mask_coord = np.ones((self.num_lms, 2))

            mask_map *= mask.reshape(self.num_lms, 1, 1)
            mask_xy *= mask.reshape(self.num_lms, 1)
            mask_coord *= mask.reshape(self.num_lms, 1)

            target_map = torch.from_numpy(target_map).float()
            target_local_x = torch.from_numpy(target_local_x).float()
            target_local_y = torch.from_numpy(target_local_y).float()
            target = torch.from_numpy(target).float().view(-1, 2)
            mask_map = torch.from_numpy(mask_map).float()
            mask_xy = torch.from_numpy(mask_xy).float()
            mask_coord = torch.from_numpy(mask_coord).float()
            target_domain = torch.from_numpy(np.array([domain])).float()

            if self.transform is not None:
                img = self.transform(img)

            return img, target_map, target_local_x, target_local_y, target, mask_map, mask_xy, mask_coord, target_domain
        ########################################################
        # for val
        else:
            if domain == 0: # source domain
                img_name = img_path.split('/')[-1]
                phy_dist = self.cfg.phy_dist
                if self.cfg.task == 'head':
                    origin_size = self.cfg.origin_size
            elif domain == 1: # target domain
                img_name = img_path.split('/')[-1]
                if self.cfg.task == 'head':
                    origin_size = self.img2size[img_name]
                    phy_dist = self.img2dist[img_name]
                else:
                    phy_dist = self.cfg.phy_dist

            target = torch.from_numpy(target).float()
            phy_dist = torch.from_numpy(np.array(phy_dist)).float()
            if self.cfg.task == 'head':
                origin_size = torch.from_numpy(np.array(origin_size)).float()

            if self.transform is not None:
                img = self.transform(img)

            if self.cfg.task == 'head':
                return img, target, origin_size, phy_dist 
            else:
                return img, target, phy_dist 
        ########################################################

    def __len__(self):
        return len(self.labels)

# loading unlabeled train data
class ImageFolder_u(data.Dataset):
    def __init__(self, cfg, phase, labels, map_size, transform=None):
        self.cfg = cfg  
        self.phase = phase  
        self.labels = labels
        self.map_size = map_size
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path = self.labels[index][0]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.labels)


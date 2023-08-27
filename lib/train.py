import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
from PIL import Image
import logging
import copy
import importlib
from math import ceil
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import * 

cudnn.deterministic = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

if not len(sys.argv) == 2:
    print('Format:')
    print('python train_file config_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

my_config = importlib.import_module(config_path, package='UDA_Med_Landmark')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

if not os.path.exists('./snapshots'):
    os.mkdir('./snapshots')
if not os.path.exists(os.path.join('./snapshots', cfg.data_name)):
    os.mkdir(os.path.join('./snapshots', cfg.data_name))
save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists('./logs'):
    os.mkdir('./logs')
if not os.path.exists(os.path.join('./logs', cfg.data_name)):
    os.mkdir(os.path.join('./logs', cfg.data_name))
log_dir = os.path.join('./logs', cfg.data_name, cfg.experiment_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

print('###########################################')
print('experiment_name:', cfg.experiment_name)
print('data_name:', cfg.data_name)
print('task:', cfg.task)
print('semi_iter:', cfg.semi_iter)
print('curriculum:', cfg.curriculum)
print('origin_size:', cfg.origin_size)
print('input_size:', cfg.input_size)
print('phy_dist:', cfg.phy_dist)
print('batch_size:', cfg.batch_size)
print('init_lr:', cfg.init_lr)
print('num_epochs:', cfg.num_epochs)
print('decay_steps:', cfg.decay_steps)
print('backbone:', cfg.backbone)
print('pretrained:', cfg.pretrained)
print('criterion_cls:', cfg.criterion_cls)
print('criterion_reg:', cfg.criterion_reg)
print('cls_loss_weight:', cfg.cls_loss_weight)
print('xy_loss_weight:', cfg.xy_loss_weight)
print('domain_loss_weight:', cfg.domain_loss_weight)
print('num_lms:', cfg.num_lms)
print('use_gpu:', cfg.use_gpu)
print('gpu_id:', cfg.gpu_id)
print('###########################################')
logging.info('###########################################')
logging.info('experiment_name: {}'.format(cfg.experiment_name))
logging.info('data_name: {}'.format(cfg.data_name))
logging.info('task: {}'.format(cfg.task))
logging.info('semi_iter: {}'.format(cfg.semi_iter))
logging.info('curriculum: {}'.format(cfg.curriculum))
logging.info('origin_size: {}'.format(cfg.origin_size))
logging.info('input_size: {}'.format(cfg.input_size))
logging.info('phy_dist: {}'.format(cfg.phy_dist))
logging.info('batch_size: {}'.format(cfg.batch_size))
logging.info('init_lr: {}'.format(cfg.init_lr))
logging.info('num_epochs: {}'.format(cfg.num_epochs))
logging.info('decay_steps: {}'.format(cfg.decay_steps))
logging.info('backbone: {}'.format(cfg.backbone))
logging.info('pretrained: {}'.format(cfg.pretrained))
logging.info('criterion_cls: {}'.format(cfg.criterion_cls))
logging.info('criterion_reg: {}'.format(cfg.criterion_reg))
logging.info('cls_loss_weight: {}'.format(cfg.cls_loss_weight))
logging.info('xy_loss_weight: {}'.format(cfg.xy_loss_weight))
logging.info('domain_loss_weight: {}'.format(cfg.domain_loss_weight))
logging.info('num_lms: {}'.format(cfg.num_lms))
logging.info('use_gpu: {}'.format(cfg.use_gpu))
logging.info('gpu_id: {}'.format(cfg.gpu_id))
logging.info('###########################################')

resnet50 = models.resnet50(pretrained=cfg.pretrained)
net = TF_resnet(resnet50, cfg)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

criterion_cls = nn.MSELoss(reduction='sum')
criterion_reg = nn.L1Loss(reduction='sum')
criterion_domain = nn.BCEWithLogitsLoss()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if cfg.task == 'head':
    labels_train_l = get_label(cfg.data_name, 'train.txt')
    labels_train_u = get_label('HeadNew', 'train.txt')
    labels_val = get_label('HeadNew', 'test.txt')
else:
    labels_train_l = get_label(cfg.data_name, 'train.txt')
    labels_train_u = get_label('MSP', 'train.txt')
    labels_val = get_label('MSP', 'test.txt')

# oversampling source domain data to make them equal
sample_times = ceil(1. * len(labels_train_u) / len(labels_train_l))
labels_train_l = labels_train_l * sample_times
labels_train_l = labels_train_l[:len(labels_train_u)]
print('oversampled labels_train_l to {}'.format(len(labels_train_l)))
logging.info('oversampled labels_train_l to {}'.format(len(labels_train_l)))

map_size = get_map_size(net, cfg.input_size, device)

data_train_u = data_utils.ImageFolder_u(cfg, 'train', 
                                        [x[:1] for x in labels_train_u],
                                        map_size,
                                        transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize]))

data_val = data_utils.ImageFolder_tf(cfg, 'val',
                                      labels_val, 
                                      map_size,
                                      transforms.Compose([
                                      transforms.ToTensor(),
                                      normalize]))

loader_train_u = torch.utils.data.DataLoader(data_train_u, batch_size=cfg.batch_size, shuffle=False, num_workers=6, pin_memory=True, drop_last=False)
loader_val = torch.utils.data.DataLoader(data_val, batch_size=cfg.batch_size, shuffle=False, num_workers=6, pin_memory=True, drop_last=False)

pseudo_labels = []
for ti in range(cfg.semi_iter):
    print('Starting iter {}'.format(ti))
    logging.info('Starting iter {}'.format(ti))

    resnet50 = models.resnet50(pretrained=cfg.pretrained)
    net = TF_resnet(resnet50, cfg)
    net = net.to(device)

    if ti == 0:
        labels_train = labels_train_l + labels_train_u
    else:
        labels_train = labels_train_l + pseudo_labels

    data_train = data_utils.ImageFolder_tf(cfg, 'train',
                                           labels_train, 
                                           map_size,
                                           transforms.Compose([
                                           transforms.ToTensor(),
                                           normalize]))

    loader_train = torch.utils.data.DataLoader(data_train, batch_size=cfg.batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=False)

    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)
    
    # train with GT source data + unlabeled/pseudo-labeled target data
    net = train_model(cfg, cfg.num_epochs, net, loader_train, loader_val, criterion_cls, criterion_reg, criterion_domain, optimizer, scheduler, os.path.join(save_dir, 'last.pth'), ti, device)

    ###############
    # test
    print('After iter {}'.format(ti))
    logging.info('After iter {}'.format(ti))
    
    mres, distances = val_model(cfg, net, loader_val, device)
    print('mre: {}'.format(np.mean(mres)))
    logging.info('mre: {}'.format(np.mean(mres)))
    sdrs = compute_sdr(distances, [2, 2.5, 3, 4])
    print('sdrs: {}'.format(sdrs))
    logging.info('sdrs: {}'.format(sdrs))
    

    ###############
    # estimate pseudo labels
    pseudo_labels = gen_pseudo(cfg, net, loader_train_u, ti, device)


import os, cv2
import numpy as np
from PIL import Image, ImageFilter
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import time
import math
import json

cudnn.deterministic = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

def get_label(data_name, label_file):
    label_path = os.path.join('data', data_name, label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        if data_name == 'Head':
            if label_file == 'train.txt':
                image_path = os.path.join('data', 'Head', 'images_train', image_name)
            else:
                image_path = os.path.join('data', 'Head', 'images_test', image_name)
            domain = 0
            mask = np.ones((len(target)//2))
        elif data_name == 'HeadNew':
            if label_file == 'train.txt':
                image_path = os.path.join('data', 'HeadNew', 'images_train', image_name)
            else:
                image_path = os.path.join('data', 'HeadNew', 'images_test', image_name)
            domain = 1
            mask = np.zeros((len(target)//2))
        elif data_name == 'JSRT':
            if label_file == 'train.txt':
                image_path = os.path.join('data', 'JSRT', 'images_train', image_name)
            else:
                image_path = os.path.join('data', 'JSRT', 'images_test', image_name)
            domain = 0
            mask = np.ones((len(target)//2))
        elif data_name == 'MSP':
            if label_file == 'train.txt':
                image_path = os.path.join('data', 'MSP', 'images_train', image_name)
            else:
                image_path = os.path.join('data', 'MSP', 'images_test', image_name)
            domain = 1
            mask = np.zeros((len(target)//2))
        else:
            print('error!')
            exit(0)
        labels_new.append([image_path, target, mask, domain])
    return labels_new

def get_map_size(net, input_size, device):
    net.eval()
    with torch.no_grad():
        data_tmp = torch.rand(1, 3, input_size[0], input_size[1])
        data_tmp = data_tmp.to(device)
        out = net(data_tmp, 0)
    if isinstance(out, tuple):
        return list(out[0].size())
    else:
        return list(out.size())

def compute_loss_tf(outputs_map, outputs_local_x, outputs_local_y, outputs, labels_map, labels_local_x, labels_local_y, labels, masks_map, masks_xy, masks_coord, criterion_cls, criterion_reg, cfg):

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
    labels_map = labels_map.view(tmp_batch*tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_map, 1)

    indices = torch.topk(labels_map, 21)[1]
    outputs_local_x = outputs_local_x.view(tmp_batch*tmp_channel, -1)
    outputs_local_x_select = torch.gather(outputs_local_x, 1, indices)
    outputs_local_y = outputs_local_y.view(tmp_batch*tmp_channel, -1)
    outputs_local_y_select = torch.gather(outputs_local_y, 1, indices)

    labels_local_x = labels_local_x.view(tmp_batch*tmp_channel, -1)
    labels_local_x_select = torch.gather(labels_local_x, 1, indices)
    labels_local_y = labels_local_y.view(tmp_batch*tmp_channel, -1)
    labels_local_y_select = torch.gather(labels_local_y, 1, indices)

    labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_map = criterion_cls(outputs_map*masks_map, labels_map*masks_map)
    if not masks_map.sum() == 0:
        loss_map /= masks_map.sum()

    masks_xy = masks_xy.view(tmp_batch*tmp_channel, -1)

    loss_x = criterion_reg(outputs_local_x_select*masks_xy, labels_local_x_select*masks_xy)
    if not masks_xy.sum() == 0:
        loss_x /= masks_xy.sum()

    loss_y = criterion_reg(outputs_local_y_select*masks_xy, labels_local_y_select*masks_xy)
    if not masks_xy.sum() == 0:
        loss_y /= masks_xy.sum()

    loss_coord = criterion_reg(outputs*masks_coord, labels*masks_coord)
    if not masks_coord.sum() == 0:
        loss_coord /= masks_coord.sum()

    return loss_map, loss_x, loss_y, loss_coord
 
def compute_loss_domain(outputs_domain, labels_domain, criterion_domain):
    loss_domain = criterion_domain(outputs_domain, labels_domain)
    return loss_domain
 
def train_model(cfg, num_epochs, net, loader_train, loader_val, criterion_cls, criterion_reg, criterion_domain, optimizer, scheduler, save_path, cur_iter, device):
    for epoch in range(num_epochs):
        coef_grl = 2/(1+np.exp(-10*(1.*epoch/num_epochs))) - 1

        net.train()
        for i, data  in enumerate(loader_train):
            inputs, labels_map, labels_x, labels_y, labels, masks_map, masks_xy, masks_coord, labels_domain = data
            inputs = inputs.to(device)
            labels_map = labels_map.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)
            labels = labels.to(device)
            masks_map = masks_map.to(device)
            masks_xy = masks_xy.to(device)
            masks_coord = masks_coord.to(device)
            labels_domain = labels_domain.to(device)
            outputs_map, outputs_x, outputs_y, outputs, outputs_domain = net(inputs, coef_grl)
            loss_domain = compute_loss_domain(outputs_domain, labels_domain, criterion_domain)
            loss_map, loss_x, loss_y, loss_coord = compute_loss_tf(outputs_map, outputs_x, outputs_y, outputs, labels_map, labels_x, labels_y, labels, masks_map, masks_xy, masks_coord, criterion_cls, criterion_reg, cfg)
            loss = cfg.cls_loss_weight*loss_map + cfg.xy_loss_weight*loss_x + cfg.xy_loss_weight*loss_y + loss_coord + cfg.domain_loss_weight*loss_domain

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <coord loss: {:.6f}> <domain loss: {:.6f}>'.format(epoch, num_epochs-1, i, len(loader_train)-1, loss.item(), cfg.cls_loss_weight*loss_map.item(), cfg.xy_loss_weight*loss_x.item(), cfg.xy_loss_weight*loss_y.item(), loss_coord.item(), cfg.domain_loss_weight*loss_domain.item()))
                logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <coord loss: {:.6f}> <domain loss: {:.6f}>'.format(epoch, num_epochs-1, i, len(loader_train)-1, loss.item(), cfg.cls_loss_weight*loss_map.item(), cfg.xy_loss_weight*loss_x.item(), cfg.xy_loss_weight*loss_y.item(), loss_coord.item(), cfg.domain_loss_weight*loss_domain.item()))

        scheduler.step()
    torch.save(net.state_dict(), save_path)
    print('saving model to {}'.format(save_path))
    logging.info('saving model to {}'.format(save_path))
    return net

def val_model(cfg, net, loader_val, device):
    mres = []
    distances = []
    for i, data in enumerate(loader_val):
        if cfg.task == 'head':
            inputs, labels, origin_sizes, phy_dists = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            origin_sizes = origin_sizes.to(device)
            phy_dists = phy_dists.to(device)
            outputs, cls_score = forward_tf(net, inputs, cfg.input_size, 4)
            tmp_batch, tmp_channel, tmp_dim = outputs.size()
            outputs = outputs.cpu().numpy()
            # batch x channel x 2
            labels = labels.cpu().numpy()
            phy_dist_new = (origin_sizes * phy_dists.view(-1,1)).cpu().numpy() / np.array(cfg.input_size)
            # swap x and y
            phy_dist_new = phy_dist_new[:, ::-1]
            phy_dist_new = np.repeat(phy_dist_new, cfg.num_lms, axis=0)
        else:
            inputs, labels, phy_dists = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            phy_dists = phy_dists.to(device)
            outputs, cls_score = forward_tf(net, inputs, cfg.input_size, 4)
            tmp_batch, tmp_channel, tmp_dim = outputs.size()
            outputs = outputs.cpu().numpy()
            # batch x channel x 2
            labels = labels.cpu().numpy()
            phy_dist_new = np.repeat(phy_dists.view(-1,1).cpu().numpy(), 2, axis=1)
            phy_dist_new = np.repeat(phy_dist_new, cfg.num_lms, axis=0)

        distance = compute_distance(outputs, labels, cfg.input_size, phy_dist_new)
        distance = distance.reshape(tmp_batch, tmp_channel)
        for d in distance:
            mres.append(np.mean(d))
            distances += d.tolist()
    return mres, distances
    
def forward_tf(net, inputs, input_size, net_stride):
    net.eval()
    with torch.no_grad():
        outputs_cls, outputs_x, outputs_y, outputs_coord, outputs_domain = net(inputs, 0)
        outputs_coord = torch.clamp(outputs_coord, min=0., max=0.99)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
        mu_x = torch.floor(outputs_coord[:,:,0]*tmp_width)
        mu_y = torch.floor(outputs_coord[:,:,1]*tmp_height)
        max_ids = (mu_y*tmp_width+mu_x).long().view(-1, 1)

        outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)
        outputs_cls_score = torch.gather(outputs_cls, 1, max_ids)
        outputs_cls_score = outputs_cls_score.view(tmp_batch, tmp_channel)

        outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)
        outputs_x_select = torch.gather(outputs_x, 1, max_ids)
        outputs_x_select = outputs_x_select.squeeze(1)
        outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)
        outputs_y_select = torch.gather(outputs_y, 1, max_ids)
        outputs_y_select = outputs_y_select.squeeze(1)

        tmp_x = mu_x.view(-1,1)+outputs_x_select.view(-1,1)
        tmp_y = mu_y.view(-1,1)+outputs_y_select.view(-1,1)
        tmp_x /= 1.0 * input_size[1] / net_stride
        tmp_y /= 1.0 * input_size[0] / net_stride

        coord = torch.cat((tmp_x.view(tmp_batch, tmp_channel, 1), tmp_y.view(tmp_batch, tmp_channel, 1)), 2)

    return coord, outputs_cls_score

def compute_distance(lms_pred, lms_gt, input_size, phy_dist):
    lms_pred = lms_pred.reshape((-1, 2))*np.array([[input_size[1],input_size[0]]])*phy_dist
    lms_gt = lms_gt.reshape((-1, 2))*np.array([[input_size[1],input_size[0]]])*phy_dist
    distance = np.linalg.norm(lms_pred - lms_gt, axis=1)
    return distance

def compute_sdr(distances, threshs=[2, 2.5, 3, 4]):
    num_total = len(distances)
    distances = np.array(distances)
    sdrs = []
    for thresh in threshs:
        num_success = np.sum((distances < thresh).astype(int))
        sdrs.append(num_success * 1. / num_total)
    return sdrs

def gen_pseudo(cfg, net, loader_train_u, cur_iter, device):
    img_paths_list = []
    outputs_list = []
    conf_scores_list = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader_train_u):
            inputs, img_paths = data
            inputs = inputs.to(device)

            outputs, cls_score = forward_tf(net, inputs, cfg.input_size, cfg.net_stride)

            img_paths_list += img_paths
            outputs_list.append(outputs)
            conf_scores_list.append(cls_score)
    outputs_list = torch.cat(outputs_list, dim=0)
    conf_scores_list = torch.cat(conf_scores_list, dim=0)
    scores_sorted, indices = torch.sort(conf_scores_list, dim=0, descending=True)
    
    perc = cfg.curriculum[cur_iter]
    num_select = int(len(img_paths_list)*perc)
    conf_thresh = scores_sorted[num_select-1, :].repeat(len(img_paths_list), 1)
    masks_list = (conf_scores_list >= conf_thresh).float()
    pseudo_labels = [[img_paths_list[i], outputs_list[i].flatten().cpu().numpy(), masks_list[i].flatten().cpu().numpy(), 1] for i in range(len(img_paths_list))]
    return pseudo_labels


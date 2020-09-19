# -*- encoding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from model import BiSeNet
from face_dataset import FaceMask
from loss import OhemCELoss
from evaluate import evaluate
import torch.optim as Optimizer
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os.path as osp
import time
import datetime
import argparse

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(fintune_model,data_root,respth):

    # dataset
    n_classes = 19
    n_img_per_gpu = 16
    n_workers = 8
    cropsize = [448, 448]

    ds = FaceMask(data_root, cropsize=cropsize, mode='train')
    # sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = n_img_per_gpu,
                    shuffle = True,
                    num_workers = n_workers,
                    pin_memory = True,
                    drop_last = True)

    # model
    ignore_idx = -100

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    net = BiSeNet(n_classes=n_classes)
    net = net.to(device)

    if os.access(fintune_model,os.F_OK) and (fintune_model is not None):# checkpoint
        chkpt = torch.load(fintune_model, map_location=device)
        net.load_state_dict(chkpt)
        print('load fintune model : {}'.format(fintune_model))

    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//16
    LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_epoch = 1000

    optim = Optimizer.SGD(
            net.parameters(),
            lr = lr_start,
            momentum = momentum,
            weight_decay = weight_decay)

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    # diter = iter(dl)
    epoch = 0
    flag_change_lr_cnt = 0 # 学习率更新计数器
    init_lr = lr_start # 学习率

    best_loss = np.inf
    loss_mean = 0. # 损失均值
    loss_idx = 0. # 损失计算计数器

    print('start training ~')
    it = 0
    for epoch in range(max_epoch):
        net.train()
        # 学习率更新策略
        if loss_mean!=0.:
            if best_loss > (loss_mean/loss_idx):
                flag_change_lr_cnt = 0
                best_loss = (loss_mean/loss_idx)
            else:
                flag_change_lr_cnt += 1

                if flag_change_lr_cnt > 30:
                    init_lr = init_lr*0.96
                    set_learning_rate(optimizer, init_lr)
                    flag_change_lr_cnt = 0

        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器

        for i, (im, lb) in enumerate(dl):

            im = im.cuda()
            lb = lb.cuda()
            H, W = im.size()[2:]
            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            out, out16, out32 = net(im)
            lossp = LossP(out, lb)
            loss2 = Loss2(out16, lb)
            loss3 = Loss3(out32, lb)
            loss = lossp + loss2 + loss3

            loss_mean += loss.item()
            loss_idx += 1.

            loss.backward()
            optim.step()


            if it % msg_iter == 0:

                print('epoch <{}/{}> -->> <{}/{}> -> iter {} : loss {:.5f}, loss_mean :{:.5f}, best_loss :{:.5f},lr :{:.6f},batch_size : {}'.\
                format(epoch,max_epoch,i,int(ds.__len__()/n_img_per_gpu),it,loss.item(),loss_mean/loss_idx,best_loss,init_lr,n_img_per_gpu))
                # print(msg)

                if (it) % 500 == 0:
                    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                    torch.save(state, respth+'/model/face_parse_latest.pth')
                    # evaluate(dspth='./images', cp='{}_iter.pth'.format(it))
            it += 1
        torch.save(state, respth+'/model/face_parse_epoch_{}.pth'.format(epoch))

if __name__ == "__main__":
    respth = './result'
    data_root = './CelebAMask-HQ/'
    if not osp.exists(respth):
        os.makedirs(respth)
    if not osp.exists(respth+'/model'):
        os.makedirs(respth+'/model')
    fintune_model = './fintune_model/79999_iter.pth'
    
    train(fintune_model,data_root,respth)

import argparse
from heapq import merge
import os
import time
import numpy as np
import pandas as pd

import shutil
import sys
import pdb
import logging

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from data_utils.dataloader import Maxillofacial3D
import data_utils.transforms as tr
from utils import setgpu, box_generate, get_iou
from models.losses import AlignLoss
from models.align import AlignNet


################################################################################

# [Train]
# python3 align.py

# [Test]
# python3 align.py --resume ./SavePath/align/model_best.ckpt --test_flag 1

################################################################################


parser = argparse.ArgumentParser(description='PyTorch DeepMarker Landmark')
parser.add_argument('--epochs',default=201, type=int)
parser.add_argument('--start_epoch',default=0,type=int)
parser.add_argument('--batch_size', default=4,type=int)
parser.add_argument('--lr_align', default=0.001,type=float)
parser.add_argument('--momentum',default=0.9,type=float)
parser.add_argument('--resume',default='',type=str)
parser.add_argument('--weight-decay','--wd',default=0.0005,type=float)
parser.add_argument('--save_dir',default='./SavePath/align/', type=str)
parser.add_argument('--gpu',default='all',type=str)
parser.add_argument('--parent_path',default='../data/',type=str)
parser.add_argument('--data_path',default='20230713CT200',type=str) # Data path
parser.add_argument('--data_type',default='ceph',type=str) # Data type (ct, cbct)
parser.add_argument('--test_flag',default=-1,type=int,help='-1 for train, 0 for eval, 1 for test')
DEVICE = torch.device("cuda" if True else "cpu")


size_align = [24, 24, 24]


def main(args):
    cudnn.benchmark = True
    setgpu(args.gpu)

    
    ##########################################        Model Init       #####################################################
    
    net_align = AlignNet(int((size_align[0] - 4) / 2 - 4) * int((size_align[1] - 4) / 2 - 4) * int((size_align[2] - 4) / 2 - 4))
    loss_align = AlignLoss()

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    logging.info(args)

    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        net_align.load_state_dict(checkpoint['state_dict_align'])

    net_align = net_align.to(DEVICE)
    loss_align = loss_align.to(DEVICE)
    if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
        net_align = DataParallel(net_align)


    ##########################################        Test       #####################################################

    if args.test_flag > -1:
        test_transform = transforms.Compose([
            tr.ZoomOut(size_align),
            tr.Normalize()
        ])
        if args.test_flag == 0:
            phase = 'val'
        else:
            phase = 'test'
        test_dataset = Maxillofacial3D(transform=test_transform,
                      phase=phase,
                      parent_path=args.parent_path,
                      data_path=args.data_path)
        testloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=0)
        test(testloader, net_align, start_epoch)
        return


    ##########################################        Dataset Prepare       #####################################################

    train_transform = transforms.Compose([
        tr.ZoomOut(size_align),
        tr.Normalize()
    ])
    train_dataset = Maxillofacial3D(transform=train_transform,
                      phase='train',
                      parent_path=args.parent_path,
                      data_path=args.data_path)
    trainloader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=4)

    eval_transform = transforms.Compose([
        tr.ZoomOut(size_align),
        tr.Normalize()
    ])
    eval_dataset = Maxillofacial3D(transform=eval_transform,
                      phase='val',
                      parent_path=args.parent_path,
                      data_path=args.data_path)
    evalloader = DataLoader(eval_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4)


    ##########################################        Optimizer Init       #####################################################

    optimizer_align = torch.optim.Adam(net_align.parameters(),
                                 lr=args.lr_align,
                                 betas=(0.9, 0.98),
                                 weight_decay=args.weight_decay)
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        optimizer_align.load_state_dict(checkpoint['optimizer_align']),
    
    scheduler_align = optim.lr_scheduler.ExponentialLR(optimizer_align, gamma=0.98, last_epoch=-1)
    

    ##########################################        Train       #####################################################

    break_flag = 0.
    low_loss = 999.
    for epoch in range(start_epoch, args.epochs):
        train_total_loss_align = train(trainloader, net_align, loss_align, epoch, optimizer_align)

        eval_total_loss_align = evaluation(evalloader, net_align, loss_align, epoch)
        
        # Lr Update
        if eval_total_loss_align > low_loss:
            if optimizer_align.param_groups[0]['lr'] > args.lr_align * 0.03:
                scheduler_align.step()
        else:
            low_loss = eval_total_loss_align

        # Save Model
        if epoch % 10 == 0:
            if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
                state_dict_align = net_align.module.state_dict()
            else:
                state_dict_align = net_align.state_dict()
            checkpoint_dict = {
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict_align': state_dict_align,
                    'optimizer_align': optimizer_align.state_dict(),
                    'args': args
                }
            torch.save(checkpoint_dict, os.path.join(save_dir, 'model_{}.ckpt'.format(epoch)))
            logging.info(
                '************************ model saved successful ************************** !'
            )

        if eval_total_loss_align <= low_loss:
            if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
                state_dict_align = net_align.module.state_dict()
            else:
                state_dict_align = net_align.state_dict()
            checkpoint_dict = {
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict_align': state_dict_align,
                    'optimizer_align': optimizer_align.state_dict(),
                    'args': args
                }
            torch.save(checkpoint_dict, os.path.join(save_dir, 'model_best.ckpt'))
            logging.info(
                '************************ model_best saved successful ************************** !'
            )


def train(data_loader, net_align, loss_align, epoch, optimizer_align):
    start_time = time.time()
    net_align.train()
    total_loss_align = []
    total_iou = []

    for i, sample in enumerate(data_loader):
        print("\rTrain : {} / {}...".format(i, len(data_loader)), end="")
        # sample = {'image', 'landmark', 'spacing', 'oimage_shape', 'filename'}
        

        ##########################################        Align       #####################################################
        
        data = sample['image'].to(DEVICE) # [batch_size, 1, al_x, al_y, al_z]

        # Align: Forward
        predict_box = net_align(data) # [batch_size, 6]
        gt_box = torch.Tensor(box_generate(sample['landmark'].numpy(), sample['oimage_shape'].numpy(), pad_size=0.2)).to(DEVICE)
        cur_loss_align = loss_align(predict_box, gt_box)
        total_loss_align.append(cur_loss_align.item())
        
        # Align: Backward
        optimizer_align.zero_grad()
        cur_loss_align.backward()
        optimizer_align.step()

        # Align: IOU
        iou = get_iou(predict_box.cpu().detach().numpy(), gt_box.cpu().numpy())
        total_iou.append(iou)

    ##########################################        Summary       #####################################################

    total_iou = np.concatenate(total_iou, 0)

    total_mean_iou = np.mean(total_iou)
    # Log
    logging.info(
        'Train--Epoch[%d], total_loss_align: [%.6f], lr_align[%.6f], time: %.1f s!'
        % (epoch, np.mean(total_loss_align), optimizer_align.param_groups[0]['lr'], time.time() - start_time))
    logging.info('Train Align-- IOU: [%.4f].' %(total_mean_iou))
    return np.mean(total_loss_align)


def evaluation(data_loader, net_align, loss_align, epoch):
    start_time = time.time()
    net_align.eval()
    total_loss_align = []
    total_iou = []

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            print("\rEval : {} / {}...".format(i, len(data_loader)), end="")
            # sample = {'image', 'landmark', 'spacing', 'oimage_shape', 'filename'}


            ##########################################        Align       #####################################################

            data = sample['image'].to(DEVICE) # [batch_size, 1, al_x, al_y, al_z]

            # Align: Forward
            predict_box = net_align(data) # [batch_size, 6]
            gt_box = torch.Tensor(box_generate(sample['landmark'].numpy(), sample['oimage_shape'].numpy(), pad_size=0.2)).to(DEVICE) # [batch_size, 6]
            cur_loss_align = loss_align(predict_box, gt_box)
            total_loss_align.append(cur_loss_align.item())

            # Align: IOU
            iou = get_iou(predict_box.cpu().numpy(), gt_box.cpu().numpy())
            total_iou.append(iou)


    ##########################################        Summary       #####################################################

    total_iou = np.concatenate(total_iou, 0)

    total_mean_iou = np.mean(total_iou)

    # Log
    logging.info(
        'Eval--Epoch[%d], total_loss_align: [%.6f], time: %.1f s!'
        % (epoch, np.mean(total_loss_align), time.time() - start_time))
    logging.info('Eval Align-- IOU: [%.4f].' %(total_mean_iou))
    logging.info(
        '***************************************************************************'
    )
    return np.mean(total_loss_align)


def test(data_loader, net_align, epoch):
    start_time = time.time()
    net_align.eval()
    total_iou = []

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            print("\rTest : {} / {}...".format(i, len(data_loader)), end="")
            # sample = {'image', 'landmark', 'spacing', 'oimage_shape', 'filename'}


            ##########################################        Align       #####################################################

            data = sample['image'].to(DEVICE) # [batch_size, 1, al_x, al_y, al_z]

            # Align: Forward
            predict_box = net_align(data) # [batch_size, 6]
            gt_box = torch.Tensor(box_generate(sample['landmark'].numpy(), sample['oimage_shape'].numpy(), pad_size=0.2)).to(DEVICE) # [batch_size, 6]

            # Align: IOU
            iou = get_iou(predict_box.cpu().numpy(), gt_box.cpu().numpy())
            total_iou.append(iou)


    ##########################################        Summary       #####################################################

    total_iou = np.concatenate(total_iou, 0)

    total_mean_iou = np.mean(total_iou)

    logging.info('Test Align-- IOU: [%.4f].' %(total_mean_iou))
    logging.info(
        '***************************************************************************'
    )
    

def local(sec, what):
    return time.localtime(time.time())


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.Formatter.converter = local
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s,%(lineno)d: %(message)s\n',
                        datefmt='%Y-%m-%d(%a)%H:%M:%S',
                        filename=os.path.join(args.save_dir, 'log.txt'),
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    main(args)

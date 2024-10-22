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
import traceback

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from data_utils.dataloader import Maxillofacial3D
import data_utils.transforms as tr
from utils import setgpu, box_generate, box_crop_zoomout, patch_crop, get_iou, metric, classify_metric, get_landmark_of_original_img
from data_utils.transforms import LandMarkToMaskGaussianHeatMap
from models.losses import HNM_heatmap
from models.align import AlignNet
from models.UNet import FusionGlobalUNet3D, LocalUNet3D


################################################################################

# [Train]
# python3 main.py

# [Test]
# python3 main.py --resume ./SavePath/main/UNet3D/model_best.ckpt --test_flag 1

################################################################################


parser = argparse.ArgumentParser(description='PyTorch Oral and Maxillofacial Landmark Detection')
parser.add_argument('--FE_name', default='UNet3D',help='model_name')
parser.add_argument('--epochs',default=401, type=int)
parser.add_argument('--start_epoch',default=0,type=int)
parser.add_argument('--batch_size', default=4,type=int)
parser.add_argument('--batch_size_stage2', default=4,type=int)
parser.add_argument('--lr', default=0.001,type=float)
parser.add_argument('--momentum',default=0.9,type=float)
parser.add_argument('--align_resume',default='./SavePath/align/model_best.ckpt',type=str)
parser.add_argument('--resume',default='',type=str)
parser.add_argument('--weight-decay','--wd',default=0.0005,type=float)
parser.add_argument('--save_dir',default='./SavePath/main/', type=str)
parser.add_argument('--gpu',default='all',type=str)
parser.add_argument('--parent_path',default='../data/',type=str)
parser.add_argument('--data_path',default='20230713CT200',type=str) # Data path
parser.add_argument('--n_class',default=40,type=int, help='number of landmarks') # Landmark number
parser.add_argument('--data_type',default='ct',type=str) # Data type (ct, cbct)
parser.add_argument('--untrain_stage2_epoch',default=150,type=int)
parser.add_argument('--untest_epoch',default=100,type=int)
parser.add_argument('--loss_name',default='HNM_heatmap',type=str)
parser.add_argument('--focus_radius', default=15,type=int)
parser.add_argument('--focus_radius_stage2', default=5,type=int)
parser.add_argument('--test_flag',default=-1,type=int,help='-1 for train, 0 for eval, 1 for test')
DEVICE = torch.device("cuda" if True else "cpu")


size_align = [24, 24, 24]
size_stage1 = [128, 128, 64]
size_stage2 = [32, 32, 16]


def main(args):
    cudnn.benchmark = True
    setgpu(args.gpu)

    
    ##########################################        Model Init       #####################################################
    
    net_align = AlignNet(int((size_align[0] - 4) / 2 - 4) * int((size_align[1] - 4) / 2 - 4) * int((size_align[2] - 4) / 2 - 4))
    net_stage1 = FusionGlobalUNet3D(n_class=args.n_class, fnum=16)
    net_stage2 = LocalUNet3D(n_class=args.n_class, in_channels=64, fnum=64)
    loss_stage1 = globals()[args.loss_name](R=args.focus_radius)
    loss_stage2 = globals()[args.loss_name](R=args.focus_radius_stage2)

    start_epoch = args.start_epoch
    save_dir = args.save_dir
    logging.info(args)

    checkpoint = torch.load(args.align_resume)
    net_align.load_state_dict(checkpoint['state_dict_align'])
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        net_stage1.load_state_dict(checkpoint['state_dict_stage1'])
        net_stage2.load_state_dict(checkpoint['state_dict_stage2'])

    net_align = net_align.to(DEVICE)
    net_stage1 = net_stage1.to(DEVICE)
    net_stage2 = net_stage2.to(DEVICE)
    loss_stage1 = loss_stage1.to(DEVICE)
    loss_stage2 = loss_stage2.to(DEVICE)
    if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
        net_align = DataParallel(net_align)
        net_stage1 = DataParallel(net_stage1)
        net_stage2 = DataParallel(net_stage2)


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
        test(testloader, net_align, net_stage1, net_stage2, args.data_type, start_epoch)
        return


    ##########################################        Heatmap Generate       #####################################################

    l2h_stage1 = LandMarkToMaskGaussianHeatMap(R=args.focus_radius, 
                                    n_class=args.n_class,
                                    GPU=DEVICE,
                                    img_size=size_stage1)
    l2h_stage2 = LandMarkToMaskGaussianHeatMap(R=args.focus_radius_stage2, 
                                    n_class=1,
                                    GPU=DEVICE,
                                    img_size=size_stage2)


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
                             num_workers=0)

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
                            num_workers=0)


    ##########################################        Optimizer Init       #####################################################

    optimizer = torch.optim.Adam([{'params': net_stage1.parameters()}, {'params': net_stage2.parameters()}],
                                 lr=args.lr,
                                 betas=(0.9, 0.98),
                                 weight_decay=args.weight_decay)
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer']),
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)
    

    ##########################################        Train       #####################################################

    break_flag = 0.
    low_loss = 999.
    for epoch in range(start_epoch, args.epochs):
        train_total_loss_stage1, train_total_loss_stage2 = train(trainloader, net_align, net_stage1, net_stage2, loss_stage1, loss_stage2, epoch, l2h_stage1, l2h_stage2, optimizer)

        if epoch < args.untest_epoch:
            continue
        break_flag += 1

        eval_total_loss_stage1, eval_total_loss_stage2 = evaluation(evalloader, net_align, net_stage1, net_stage2, loss_stage1, loss_stage2, epoch, l2h_stage1, l2h_stage2)

        # Lr Update
        if epoch >= args.untrain_stage2_epoch:
            if eval_total_loss_stage2 > low_loss:
                if optimizer.param_groups[0]['lr'] > args.lr * 0.03:
                    scheduler.step()
            else:
                break_flag = 0
                low_loss = eval_total_loss_stage2
        else:
            break_flag = 0

        # Save Model
        if epoch % 10 == 0:
            if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
                state_dict_stage1 = net_stage1.module.state_dict()
                state_dict_stage2 = net_stage2.module.state_dict()
            else:
                state_dict_stage1 = net_stage1.state_dict()
                state_dict_stage2 = net_stage2.state_dict()
            checkpoint_dict = {
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict_stage1': state_dict_stage1,
                    'state_dict_stage2': state_dict_stage2,
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }
            torch.save(checkpoint_dict, os.path.join(save_dir, 'model_{}.ckpt'.format(epoch)))
            logging.info(
                '************************ model saved successful ************************** !'
            )

        if eval_total_loss_stage2 <= low_loss:
            if len(args.gpu.split(',')) > 1 or args.gpu == 'all':
                state_dict_stage1 = net_stage1.module.state_dict()
                state_dict_stage2 = net_stage2.module.state_dict()
            else:
                state_dict_stage1 = net_stage1.state_dict()
                state_dict_stage2 = net_stage2.state_dict()
            checkpoint_dict = {
                    'epoch': epoch,
                    'save_dir': save_dir,
                    'state_dict_stage1': state_dict_stage1,
                    'state_dict_stage2': state_dict_stage2,
                    'optimizer': optimizer.state_dict(),
                    'args': args
                }
            torch.save(checkpoint_dict, os.path.join(save_dir, 'model_best.ckpt'))
            logging.info(
                '************************ model_best saved successful ************************** !'
            )


def train(data_loader, net_align, net_stage1, net_stage2, loss_stage1, loss_stage2, epoch, l2h_stage1, l2h_stage2, optimizer):
    start_time = time.time()
    net_align.eval()
    net_stage1.train()
    net_stage2.train()
    total_loss_stage1 = []
    total_loss_stage2 = []
    total_iou = []
    total_mre_stage1 = []
    total_mre = []

    for i, sample in enumerate(data_loader):
        print("\rTrain : {} / {}...".format(i, len(data_loader)), end="")
        # sample = {'image', 'landmark', 'spacing', 'oimage_shape', 'filename'}
        

        ##########################################        Align       #####################################################

        data = sample['image'].to(DEVICE) # [batch_size, 1, al_x, al_y, al_z]

        # Align: Forward
        predict_box = net_align(data) # [batch_size, 6]
        gt_box = torch.Tensor(box_generate(sample['landmark'].numpy(), sample['oimage_shape'].numpy(), pad_size=0.2, rand=True)).to(DEVICE)

        # Align: IOU
        iou = get_iou(predict_box.cpu().detach().numpy(), gt_box.cpu().numpy())
        total_iou.append(iou)


        ##########################################        Stage1       #####################################################
        
        # Stage1: Data Crop with Box
        data_stage1 = []
        landmark_stage1 = []
        zoom_rate_stage1 = []
        crop_begin_stage1 = []
        for j in range(len(sample['filename'])):
            d_s1, lm_s1, rzr_s1, cb_s1 = box_crop_zoomout(sample['filename'][j], sample['landmark'][j].numpy(), gt_box[j].cpu().numpy(), size_stage1=size_stage1)
            data_stage1.append(d_s1)
            landmark_stage1.append(lm_s1)
            zoom_rate_stage1.append(rzr_s1)
            crop_begin_stage1.append(cb_s1)
        data_stage1 = torch.Tensor(np.stack(data_stage1))
        landmark_stage1 = np.stack(landmark_stage1)
        zoom_rate_stage1 = np.stack(zoom_rate_stage1)
        crop_begin_stage1 = np.stack(crop_begin_stage1)
        data_stage1 = data_stage1.unsqueeze(1).to(DEVICE) # [batch_size, 1, s1_x, s1_y, s1_z]

        # Stage1: Forward
        predict_heatmap_stage1, encoders_features, global_feature = net_stage1(data_stage1) # [batch_size, n_class, s1_x, s1_y, s1_z]
        global_feature = global_feature.reshape([-1] + list(global_feature.shape[2:]))
        gt_heatmap_stage1 = l2h_stage1(landmark_stage1)
        cur_loss_stage1 = loss_stage1(predict_heatmap_stage1, gt_heatmap_stage1)
        total_loss_stage1.append(cur_loss_stage1.item())

        # Stage1: Landmark
        predict_landmark_stage1 = get_landmark_of_original_img( # [batch_size, n_class, 3]
            predict_heatmap_stage1.cpu().detach().numpy(), zoom_rate_stage1, crop_begin_stage1, k_vote=int(np.pi * args.focus_radius ** 2))

        # Stage1: Metric
        mre, pred_num, target_num, hits = metric(predict_landmark_stage1, sample['spacing'].numpy(), sample['landmark'].numpy())
        total_mre_stage1.append(mre)


        ##########################################        Stage2       #####################################################

        if epoch >= args.untrain_stage2_epoch:
            # Stage2: Data Prepare(Crop and Landmark Update)
            data_stage2 = []
            landmark_stage2 = []
            crop_begin_stage2 = []

            for j in range(len(sample['filename'])):
                d_s2, lm_s2, cb_s2 = patch_crop(sample['filename'][j], predict_landmark_stage1[j], sample['landmark'][j], size_stage2=size_stage2)
                data_stage2.append(d_s2)
                landmark_stage2.append(lm_s2)
                crop_begin_stage2.append(cb_s2)
            data_stage2 = torch.Tensor(np.expand_dims(np.concatenate(data_stage2), 1)) # [batch_size*n_class, 1, s2_x, s2_y, s2_z]
            landmark_stage2 = np.expand_dims(np.concatenate(landmark_stage2), 1) # [batch_size*n_class, 1, 3]
            gt_heatmap_stage2 = l2h_stage2(landmark_stage2) # [batch_size*n_class, 1, s2_x, s2_y, s2_z]
            crop_begin_stage2 = np.concatenate(crop_begin_stage2)

            # Stage2: Forward and Landmark
            cur_loss_stage2 = 0
            predict_landmarks_stage2 = np.zeros([data_stage2.shape[0], 1, 3])
            for j in range(0, data_stage2.shape[0], args.batch_size_stage2):
                # Stage2: Forward
                data_batch_stage2 = data_stage2[j: min(j + args.batch_size_stage2, data_stage2.shape[0])].to(DEVICE) # [batch_size_stage2, 1, s2_x, s2_y, s2_z]
                batch_size_stage2 = data_batch_stage2.shape[0]
                global_feature_batch_stage2 = global_feature[j: min(j + batch_size_stage2, data_stage2.shape[0])]
                data_batch_stage2 = torch.cat([data_batch_stage2, global_feature_batch_stage2], 1)
                predict_heatmap_batch_stage2 = net_stage2(data_batch_stage2) # [batch_size_stage2, n_class, s2_x, s2_y, s2_z]
                temp_heatmap = []
                for k in range(batch_size_stage2):
                    heatmap_class_id = (j + k) % sample['landmark'].shape[1]
                    temp_heatmap.append(predict_heatmap_batch_stage2[k, heatmap_class_id: heatmap_class_id + 1])
                predict_heatmap_batch_stage2 = torch.stack(temp_heatmap)
                gt_heatmap_batch_stage2 = gt_heatmap_stage2[j: min(j + batch_size_stage2, data_stage2.shape[0])] # [batch_size_stage2, 1, s2_x, s2_y, s2_z]
                cur_loss_stage2 += loss_stage2(predict_heatmap_batch_stage2, gt_heatmap_batch_stage2) * data_batch_stage2.shape[0]

                # Stage2: Landmark
                crop_begin_batch = crop_begin_stage2[j: min(j + batch_size_stage2, data_stage2.shape[0])]  # [batch_size_stage2, 3]
                zoom_rate_stage2 = np.ones([batch_size_stage2, 3])
                predict_landmarks_stage2[j: min(j + batch_size_stage2, data_stage2.shape[0])] = get_landmark_of_original_img(
                    predict_heatmap_batch_stage2.cpu().detach().numpy(), zoom_rate_stage2, crop_begin_batch, k_vote=int(np.pi * args.focus_radius_stage2 ** 2))
            cur_loss_stage2 = cur_loss_stage2 / data_stage2.shape[0]
            alpha = max(0.1, args.untrain_stage2_epoch / 100 + 1 - epoch / 100) # epoch=s2epoch: 1, epoch=max: 0.1
            cur_loss = alpha * cur_loss_stage1 + cur_loss_stage2
            total_loss_stage2.append(cur_loss_stage2.item())
            predict_landmarks_stage2 = predict_landmarks_stage2.reshape(sample['landmark'].shape)
            
            # Stage2: Metric
            mre, pred_num, target_num, hits = metric(predict_landmarks_stage2, sample['spacing'].numpy(), sample['landmark'].numpy())
        else:
            cur_loss = cur_loss_stage1
            total_loss_stage2.append(cur_loss_stage1.item())

        # Backward
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        total_mre.append(mre)


    ##########################################        Summary       #####################################################

    total_iou = np.concatenate(total_iou, 0)
    total_mre_stage1 = np.concatenate(total_mre_stage1, 0)
    total_mre_stage1 = total_mre_stage1[total_mre_stage1 >= 0]
    total_mre = np.concatenate(total_mre, 0)
    total_mre = total_mre[total_mre >= 0]

    total_mean_iou = np.mean(total_iou)

    total_mean_mre_stage1 = np.mean(total_mre_stage1)
    total_sd_stage1 = np.sqrt(np.sum(pow(total_mre_stage1 - total_mean_mre_stage1, 2)) / total_mre_stage1.shape[0])

    total_mean_mre = np.mean(total_mre)
    total_sd = np.sqrt(np.sum(pow(total_mre - total_mean_mre, 2)) / total_mre.shape[0])

    # Log
    logging.info(
        'Train--Epoch[%d], total loss Stage1: [%.6f], total loss Stage2: [%.6f], lr[%.6f], time: %.1f s!'
        % (epoch, np.mean(total_loss_stage1), np.mean(total_loss_stage2), optimizer.param_groups[0]['lr'], time.time() - start_time))
    logging.info('Train Align-- IOU: [%.4f].' %(total_mean_iou))
    logging.info('Train Stage1-- MRE: [%.2f] + SD: [%.2f].' %(total_mean_mre_stage1, total_sd_stage1))
    logging.info('Train       -- MRE: [%.2f] + SD: [%.2f].' %(total_mean_mre, total_sd))
    return np.mean(total_loss_stage1), np.mean(total_loss_stage2)


def evaluation(data_loader, net_align, net_stage1, net_stage2, loss_stage1, loss_stage2, epoch, l2h_stage1, l2h_stage2):
    start_time = time.time()
    net_align.eval()
    net_stage1.eval()
    net_stage2.eval()
    total_loss_stage1 = []
    total_loss_stage2 = []
    total_iou = []
    total_mre_stage1 = []
    total_mre = []

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            print("\rEval : {} / {}...".format(i, len(data_loader)), end="")
            # sample = {'image', 'landmark', 'spacing', 'oimage_shape', 'filename'}

            ##########################################        Align       #####################################################

            data = sample['image'].to(DEVICE) # [batch_size, 1, al_x, al_y, al_z]

            # Align: Forward
            predict_box = net_align(data) # [batch_size, 6]
            gt_box = torch.Tensor(box_generate(sample['landmark'].numpy(), sample['oimage_shape'].numpy(), pad_size=0.2, rand=False)).to(DEVICE)

            # Align: IOU
            iou = get_iou(predict_box.cpu().detach().numpy(), gt_box.cpu().numpy())
            total_iou.append(iou)


            ##########################################        Stage1       #####################################################
            
            # Stage1: Data Crop with Box
            data_stage1 = []
            landmark_stage1 = []
            zoom_rate_stage1 = []
            crop_begin_stage1 = []
            for j in range(len(sample['filename'])):
                d_s1, lm_s1, rzr_s1, cb_s1 = box_crop_zoomout(sample['filename'][j], sample['landmark'][j].numpy(), predict_box[j].cpu().numpy(), size_stage1=size_stage1)
                data_stage1.append(d_s1)
                landmark_stage1.append(lm_s1)
                zoom_rate_stage1.append(rzr_s1)
                crop_begin_stage1.append(cb_s1)
            data_stage1 = torch.Tensor(np.stack(data_stage1))
            landmark_stage1 = np.stack(landmark_stage1)
            zoom_rate_stage1 = np.stack(zoom_rate_stage1)
            crop_begin_stage1 = np.stack(crop_begin_stage1)
            data_stage1 = data_stage1.unsqueeze(1).to(DEVICE) # [batch_size, 1, s1_x, s1_y, s1_z]

            # Stage1: Forward
            predict_heatmap_stage1, encoders_features, global_feature = net_stage1(data_stage1) # [batch_size, n_class, s1_x, s1_y, s1_z]
            global_feature = global_feature.reshape([-1] + list(global_feature.shape[2:]))
            gt_heatmap_stage1 = l2h_stage1(landmark_stage1)
            cur_loss_stage1 = loss_stage1(predict_heatmap_stage1, gt_heatmap_stage1)
            total_loss_stage1.append(cur_loss_stage1.item())

            # Stage1: Landmark
            predict_landmark_stage1 = get_landmark_of_original_img( # [batch_size, n_class, 3]
                predict_heatmap_stage1.cpu().detach().numpy(), zoom_rate_stage1, crop_begin_stage1, k_vote=int(np.pi * args.focus_radius ** 2))

            # Stage1: Metric
            mre, pred_num, target_num, hits = metric(predict_landmark_stage1, sample['spacing'].numpy(), sample['landmark'].numpy())
            total_mre_stage1.append(mre)


            ##########################################        Stage2       #####################################################

            if epoch >= args.untrain_stage2_epoch:
                # Stage2: Data Prepare(Crop and Landmark Update)
                data_stage2 = []
                landmark_stage2 = []
                crop_begin_stage2 = []

                for j in range(len(sample['filename'])):
                    d_s2, lm_s2, cb_s2 = patch_crop(sample['filename'][j], predict_landmark_stage1[j], sample['landmark'][j], size_stage2=size_stage2)
                    data_stage2.append(d_s2)
                    landmark_stage2.append(lm_s2)
                    crop_begin_stage2.append(cb_s2)
                data_stage2 = torch.Tensor(np.expand_dims(np.concatenate(data_stage2), 1)) # [batch_size*n_class, 1, s2_x, s2_y, s2_z]
                landmark_stage2 = np.expand_dims(np.concatenate(landmark_stage2), 1) # [batch_size*n_class, 1, 3]
                gt_heatmap_stage2 = l2h_stage2(landmark_stage2) # [batch_size*n_class, 1, s2_x, s2_y, s2_z]
                crop_begin_stage2 = np.concatenate(crop_begin_stage2)

                # Stage2: Forward and Landmark
                cur_loss_stage2 = 0
                predict_landmarks_stage2 = np.zeros([data_stage2.shape[0], 1, 3])
                for j in range(0, data_stage2.shape[0], args.batch_size_stage2):
                    # Stage2: Forward
                    data_batch_stage2 = data_stage2[j: min(j + args.batch_size_stage2, data_stage2.shape[0])].to(DEVICE) # [batch_size_stage2, 1, s2_x, s2_y, s2_z]
                    batch_size_stage2 = data_batch_stage2.shape[0]
                    global_feature_batch_stage2 = global_feature[j: min(j + batch_size_stage2, data_stage2.shape[0])]
                    data_batch_stage2 = torch.cat([data_batch_stage2, global_feature_batch_stage2], 1)
                    predict_heatmap_batch_stage2 = net_stage2(data_batch_stage2) # [batch_size_stage2, n_class, s2_x, s2_y, s2_z]
                    temp_heatmap = []
                    for k in range(batch_size_stage2):
                        heatmap_class_id = (j + k) % sample['landmark'].shape[1]
                        temp_heatmap.append(predict_heatmap_batch_stage2[k, heatmap_class_id: heatmap_class_id + 1])
                    predict_heatmap_batch_stage2 = torch.stack(temp_heatmap)
                    gt_heatmap_batch_stage2 = gt_heatmap_stage2[j: min(j + batch_size_stage2, data_stage2.shape[0])] # [batch_size_stage2, 1, s2_x, s2_y, s2_z]
                    cur_loss_stage2 += loss_stage2(predict_heatmap_batch_stage2, gt_heatmap_batch_stage2) * data_batch_stage2.shape[0]

                    # Stage2: Landmark
                    crop_begin_batch = crop_begin_stage2[j: min(j + batch_size_stage2, data_stage2.shape[0])]  # [batch_size_stage2, 3]
                    zoom_rate_stage2 = np.ones([batch_size_stage2, 3])
                    predict_landmarks_stage2[j: min(j + batch_size_stage2, data_stage2.shape[0])] = get_landmark_of_original_img(
                        predict_heatmap_batch_stage2.cpu().detach().numpy(), zoom_rate_stage2, crop_begin_batch, k_vote=int(np.pi * args.focus_radius_stage2 ** 2))
                cur_loss_stage2 = cur_loss_stage2 / data_stage2.shape[0]
                total_loss_stage2.append(cur_loss_stage2.item())
                predict_landmarks_stage2 = predict_landmarks_stage2.reshape(sample['landmark'].shape)
                
                # Stage2: Metric
                mre, pred_num, target_num, hits = metric(predict_landmarks_stage2, sample['spacing'].numpy(), sample['landmark'].numpy())
            else:
                total_loss_stage2.append(cur_loss_stage1.item())


            total_mre.append(mre)


    ##########################################        Summary       #####################################################

    total_iou = np.concatenate(total_iou, 0)
    total_mre_stage1 = np.concatenate(total_mre_stage1, 0)
    total_mre_stage1 = total_mre_stage1[total_mre_stage1 >= 0]
    total_mre = np.concatenate(total_mre, 0)
    total_mre = total_mre[total_mre >= 0]

    total_mean_iou = np.mean(total_iou)

    total_mean_mre_stage1 = np.mean(total_mre_stage1)
    total_sd_stage1 = np.sqrt(np.sum(pow(total_mre_stage1 - total_mean_mre_stage1, 2)) / total_mre_stage1.shape[0])

    total_mean_mre = np.mean(total_mre)
    total_sd = np.sqrt(np.sum(pow(total_mre - total_mean_mre, 2)) / total_mre.shape[0])
    
    # Log
    logging.info(
        'Eval--Epoch[%d], total loss Stage1: [%.6f], total loss Stage2: [%.6f], time: %.1f s!'
        % (epoch, np.mean(total_loss_stage1), np.mean(total_loss_stage2), time.time() - start_time))
    logging.info('Eval Align-- IOU: [%.4f].' %(total_mean_iou))
    logging.info('Eval Stage1-- MRE: [%.2f] + SD: [%.2f].' %(total_mean_mre_stage1, total_sd_stage1))
    logging.info('Eval       -- MRE: [%.2f] + SD: [%.2f].' %(total_mean_mre, total_sd))
    logging.info(
        '***************************************************************************'
    )
    return np.mean(total_loss_stage1), np.mean(total_loss_stage2)


def test(data_loader, net_align, net_stage1, net_stage2, data_type, epoch):
    start_time = time.time()
    net_align.eval()
    net_stage1.eval()
    net_stage2.eval()
    total_iou = []
    total_mre_stage1 = []
    total_mre = []
    total_pred = []
    total_target = []
    total_lm = []
    if data_type=="ceph":
        total_hits_stage1 = np.zeros((8, 40))
        total_hits = np.zeros((8, 40))
    elif data_type=="cbct":
        total_hits_stage1 = np.zeros((8, 13))
        total_hits = np.zeros((8, 13))
    elif data_type=="pddca":
        total_hits_stage1 = np.zeros((8, 5))
        total_hits = np.zeros((8, 5))
    elif data_type=="molar":
        total_hits_stage1 = np.zeros((8, 14))
        total_hits = np.zeros((8, 14))
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            print("\rTest : {} / {}...".format(i, len(data_loader)), end="")
            # sample = {'image', 'landmark', 'spacing', 'oimage_shape', 'filename'}
            

            ##########################################        Align       #####################################################

            data = sample['image'].to(DEVICE) # [batch_size, 1, al_x, al_y, al_z]

            # Align: Forward
            predict_box = net_align(data) # [batch_size, 6]
            gt_box = torch.Tensor(box_generate(sample['landmark'].numpy(), sample['oimage_shape'].numpy(), pad_size=0.2, rand=False)).to(DEVICE)

            # Align: IOU
            iou = get_iou(predict_box.cpu().detach().numpy(), gt_box.cpu().numpy())
            total_iou.append(iou)


            ##########################################        Stage1       #####################################################
            
            # Stage1: Data Crop with Box
            data_stage1 = []
            landmark_stage1 = []
            zoom_rate_stage1 = []
            crop_begin_stage1 = []
            for j in range(len(sample['filename'])):
                d_s1, lm_s1, rzr_s1, cb_s1 = box_crop_zoomout(sample['filename'][j], sample['landmark'][j].numpy(), predict_box[j].cpu().numpy(), size_stage1=size_stage1)
                data_stage1.append(d_s1)
                landmark_stage1.append(lm_s1)
                zoom_rate_stage1.append(rzr_s1)
                crop_begin_stage1.append(cb_s1)
            data_stage1 = torch.Tensor(np.stack(data_stage1))
            landmark_stage1 = np.stack(landmark_stage1)
            zoom_rate_stage1 = np.stack(zoom_rate_stage1)
            crop_begin_stage1 = np.stack(crop_begin_stage1)
            data_stage1 = data_stage1.unsqueeze(1).to(DEVICE) # [batch_size, 1, s1_x, s1_y, s1_z]

            # Stage1: Forward
            predict_heatmap_stage1, encoders_features, global_feature = net_stage1(data_stage1) # [batch_size, n_class, s1_x, s1_y, s1_z]
            global_feature = global_feature.reshape([-1] + list(global_feature.shape[2:]))

            # Stage1: Landmark
            predict_landmark_stage1 = get_landmark_of_original_img( # [batch_size, n_class, 3]
                predict_heatmap_stage1.cpu().detach().numpy(), zoom_rate_stage1, crop_begin_stage1, k_vote=int(np.pi * args.focus_radius ** 2))

            # Stage1: Metric
            mre, pred_num, target_num, hits = metric(predict_landmark_stage1, sample['spacing'].numpy(), sample['landmark'].numpy())
            total_mre_stage1.append(mre)
            total_hits_stage1 += hits


            ##########################################        Stage2       #####################################################

            if epoch >= args.untrain_stage2_epoch:
                # Stage2: Data Prepare(Crop and Landmark Update)
                data_stage2 = []
                crop_begin_stage2 = []

                for j in range(len(sample['filename'])):
                    d_s2, lm_s2, cb_s2 = patch_crop(sample['filename'][j], predict_landmark_stage1[j], sample['landmark'][j], size_stage2=size_stage2)
                    data_stage2.append(d_s2)
                    crop_begin_stage2.append(cb_s2)
                data_stage2 = torch.Tensor(np.expand_dims(np.concatenate(data_stage2), 1)) # [batch_size*n_class, 1, s2_x, s2_y, s2_z]
                crop_begin_stage2 = np.concatenate(crop_begin_stage2)

                # Stage2: Forward and Landmark
                predict_landmarks_stage2 = np.zeros([data_stage2.shape[0], 1, 3])
                for j in range(0, data_stage2.shape[0], args.batch_size_stage2):
                    # Stage2: Forward
                    data_batch_stage2 = data_stage2[j: min(j + args.batch_size_stage2, data_stage2.shape[0])].to(DEVICE) # [batch_size_stage2, 1, s2_x, s2_y, s2_z]
                    batch_size_stage2 = data_batch_stage2.shape[0]
                    global_feature_batch_stage2 = global_feature[j: min(j + batch_size_stage2, data_stage2.shape[0])]
                    data_batch_stage2 = torch.cat([data_batch_stage2, global_feature_batch_stage2], 1)
                    predict_heatmap_batch_stage2 = net_stage2(data_batch_stage2) # [batch_size_stage2, n_class, s2_x, s2_y, s2_z]
                    temp_heatmap = []
                    for k in range(batch_size_stage2):
                        heatmap_class_id = (j + k) % sample['landmark'].shape[1]
                        temp_heatmap.append(predict_heatmap_batch_stage2[k, heatmap_class_id: heatmap_class_id + 1])
                    predict_heatmap_batch_stage2 = torch.stack(temp_heatmap)

                    # Stage2: Landmark
                    crop_begin_batch = crop_begin_stage2[j: min(j + batch_size_stage2, data_stage2.shape[0])]  # [batch_size_stage2, 3]
                    zoom_rate_stage2 = np.ones([batch_size_stage2, 3])
                    predict_landmarks_stage2[j: min(j + batch_size_stage2, data_stage2.shape[0])] = get_landmark_of_original_img(
                        predict_heatmap_batch_stage2.cpu().detach().numpy(), zoom_rate_stage2, crop_begin_batch, k_vote=int(np.pi * args.focus_radius_stage2 ** 2))
                predict_landmarks_stage2 = predict_landmarks_stage2.reshape(sample['landmark'].shape)
                total_lm.append(predict_landmarks_stage2)
                
                # Stage2: Metric
                mre, pred_num, target_num, hits = metric(predict_landmarks_stage2, sample['spacing'].numpy(), sample['landmark'].numpy())
            else:
                pass

            total_mre.append(mre)
            total_pred.append(pred_num)
            total_target.append(target_num)
            total_hits += hits

    ##########################################        Each Class       #####################################################
    total_mre = np.concatenate(total_mre).reshape(-1, args.n_class)

    if data_type == "ct":
        names = ['A','ANS','B','Ba','CoL','CoR','FzL',
            'FzR','Gn','Gns','GoL','GoR','L6L',
            'L6R','LI','Lis','Ls','Me','Mes','MxL',
            'MxR','N','Ns','OrL','OrR','PNS','PmpL',
            'PmpR','PoL','PoR','Pog','Pogs','Prn','Si',
            'Sn','U6L','U6R','UI','ZyL','ZyR']
    elif data_type == "cbct":
        names = ['A', 'B', 'Gn', 'GoL', 'GoR',
            'L6L', 'L6R', 'LI', 'Me',
            'Pog', 'U6L', 'U6R', 'UI']
    else:
        names = [
            'L0','La', 'Lb', 'Lc', 'Ld', 'Le', 'Lf', 'R0', 'Ra','Rb','Rc','Rd','Re','Rf'
            ]
    

    IDs = ["MRE", "SD", "2.0", "2.5", "3.0", "4."]
    form = {"metric": IDs}
    mre = []
    sd = []
    total_hits_stage1 = np.sum(total_hits_stage1[:4], 1) / np.sum(total_hits_stage1[4:], 1)
    totally_hits = np.sum(total_hits[:4], 1) / np.sum(total_hits[4:], 1)
    total_hits = total_hits[:4] / total_hits[4:]
    
    for i, name in enumerate(names):
        cur_mre = []
        for j in range(total_mre.shape[0]):
            if total_mre[j,i] > 0:
                cur_mre.append(total_mre[j,i])
        cur_mre = np.array(cur_mre)
        mre.append(np.mean(cur_mre))
        sd.append(np.sqrt(np.sum(pow(cur_mre - np.mean(cur_mre), 2)) / (cur_mre.shape[0] - 1)))

    mre = np.stack(mre, 0)
    sd = np.stack(sd, 0)
    total = np.stack([mre, sd], 0)
    total = np.concatenate([total, total_hits], 0)
    for i, name in enumerate(names):
        form[name] = total[:, i]
        
    df = pd.DataFrame(form, columns = form.keys())
    df.to_excel('Output/Each_class landmark_result.xlsx', index = False, header=True)


    ##########################################        Summary       #####################################################

    total_iou = np.concatenate(total_iou, 0)
    total_mean_iou = np.mean(total_iou)

    total_mre_stage1 = np.concatenate(total_mre_stage1).reshape(-1, args.n_class)
    re_stage1 = np.array([np.mean(i[i>=0]) for i in total_mre_stage1])
    re = np.array([np.mean(i[i>=0]) for i in total_mre])

    total_mean_mre_stage1 = np.mean(re_stage1)
    total_sd_stage1 = np.sqrt(np.sum(pow(re_stage1 - total_mean_mre_stage1, 2)) / re_stage1.shape[0])

    total_mean_mre = np.mean(re)
    total_sd = np.sqrt(np.sum(pow(re - total_mean_mre, 2)) / re.shape[0])

    total_pred = np.concatenate(total_pred, 0)
    total_target = np.concatenate(total_target, 0)
    acc = np.sum(total_pred==total_target) / total_pred.shape[0]
    precision, f1 = classify_metric(total_pred, total_target)

    logging.info('Pz---2mm: [%.4f], --2.5mm: [%.4f], --3mm: [%.4f], --4mm: [%.4f]' % (total_hits_stage1[0], total_hits_stage1[1], total_hits_stage1[2], total_hits_stage1[3]))
    logging.info('Pz---2mm: [%.4f], --2.5mm: [%.4f], --3mm: [%.4f], --4mm: [%.4f]' % (totally_hits[0], totally_hits[1], totally_hits[2], totally_hits[3]))
    logging.info('Test Align-- IOU: [%.4f].' %(total_mean_iou))
    logging.info('Test Stage1-- MRE: [%.2f] + SD: [%.2f].' %(total_mean_mre_stage1, total_sd_stage1))
    logging.info(
        'Test-- MRE: [%.2f] + SD: [%.2f], ACC: [%.4f], Precision: [%.4f], F1: [%.4f], using time: %.1f s!' %(
            total_mean_mre, total_sd, 
            acc, precision, f1,
            time.time()-start_time))
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
    args.save_dir = os.path.join(args.save_dir, args.FE_name)
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

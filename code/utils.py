import os
import torch
import pdb
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import zoom
import random


def setgpu(gpus):
    if gpus=='all':
        gpus = '0,1,2,3'
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))


########################################################################################


def box_generate(landmark, oimage_shape, pad_size, rand=False):
    lm_perc = landmark / oimage_shape.reshape(-1, 1, 3)
    max_lm_perc = lm_perc.max(1)
    lm_perc[lm_perc < 0] = 1
    min_lm_perc = lm_perc.min(1)
    if rand:
        min_pad_size = pad_size - 0.05 + random.random() * 0.1
        max_pad_size = pad_size - 0.05 + random.random() * 0.1
    else:
        min_pad_size = pad_size
        max_pad_size = pad_size
    min_point = np.max(np.stack([np.zeros([lm_perc.shape[0], 3]), min_lm_perc - min_pad_size]), 0)
    max_point = np.min(np.stack([np.ones([lm_perc.shape[0], 3]), max_lm_perc + max_pad_size]), 0)
    box = np.concatenate([min_point, max_point], 1)
    return box


def box_crop_zoomout(filename, landmark, box, size_stage1=[128, 128, 64]):
    try:
        size_stage1 = np.array(size_stage1)
        img = np.load(filename)
        box = (box.reshape(2, 3) * np.array(img.shape)).astype(np.int)
        img = img[box[0, 0]: box[1, 0], box[0, 1]: box[1, 1], box[0, 2]: box[1, 2]]
        zoom_rate = size_stage1 / np.array(img.shape)
        crop_begin = box[0]
        img_pro = zoom(img, zoom_rate, order=1)
        landmark_pro = (landmark - box[0]) * zoom_rate
    except Exception as e:
        print("Error: ", filename, "Image shape: ", np.array(img.shape))
        exit(0)
    return img_pro, landmark_pro, zoom_rate, crop_begin


def patch_crop(filename, predict_landmark, landmark, size_stage2=[32, 32, 16]):
    n_class = landmark.shape[0]
    size_stage2 = np.array(size_stage2)
    img = np.load(filename)

    crop_begin = (predict_landmark - size_stage2 / 2).astype(np.int)
    crop_begin = np.max([crop_begin, np.zeros_like(crop_begin)], 0)
    crop_begin = np.min([crop_begin, np.repeat(np.expand_dims((img.shape - size_stage2), 0), repeats=n_class, axis=0)], 0)
    
    landmark_pro = landmark - crop_begin
    img_pro = []
    for i in range(n_class):
        img_crop = img[crop_begin[i, 0]:crop_begin[i, 0]+size_stage2[0], crop_begin[i, 1]:crop_begin[i, 1]+size_stage2[1], crop_begin[i, 2]:crop_begin[i, 2]+size_stage2[2]].copy()
        img_pro.append(img_crop)
    img_pro = np.stack(img_pro, 0)
    return img_pro, landmark_pro, crop_begin


########################################################################################


def get_predict_landmark(heatmap, k_vote):
    # 取热图中值最大的k_vote个体素取平均坐标作为标志点坐标
    topk = torch.Tensor(heatmap.reshape(-1)).kthvalue((heatmap.reshape(-1).shape[0] - k_vote)).values.item()
    score_idxs = np.where(heatmap >= topk)
    predict_landmark = np.array(
        [np.mean(score_idxs[0]), np.mean(score_idxs[1]), np.mean(score_idxs[2])])
    return predict_landmark


def get_landmark_of_original_img(heatmap, zoom_rate, crop_begin, k_vote=500): 
    # [batchsize, n_class, 128, 128, 64], [batchsize, 3], [batchsize, 3]
    N = heatmap.shape[0]
    n_class = heatmap.shape[1]
    predict_landmarks_batch = []
    
    for j in range(N):
        predict_landmarks = []
        for i in range(n_class):
            predict_landmark = get_predict_landmark(heatmap[j, i], k_vote=k_vote)
            predict_landmark /= zoom_rate[j]
            predict_landmark += crop_begin[j]
            predict_landmarks.append(predict_landmark)
        predict_landmarks = np.stack(predict_landmarks, 0)
        predict_landmarks_batch.append(predict_landmarks)
    predict_landmarks_batch = np.stack(predict_landmarks_batch, 0)
    return predict_landmarks_batch


########################################################################################


def get_iou(pred_box, gt_box):
    lt = np.max(np.stack([pred_box[:, :3], gt_box[:, :3]]), axis=0)
    rb = np.min(np.stack([pred_box[:, 3:], gt_box[:, 3:]]), axis=0)
    inter_volume_shape = rb - lt
    inter_volume_shape[inter_volume_shape < 0] = 0
    inter_volume = inter_volume_shape[:, 0] * inter_volume_shape[:, 1] * inter_volume_shape[:, 2]
    
    pred_volume_shape = pred_box[:, 3:] - pred_box[:, :3]
    pred_volume_shape[pred_volume_shape < 0] = 0
    pred_volume = pred_volume_shape[:, 0] * pred_volume_shape[:, 1] * pred_volume_shape[:, 2]

    gt_volume_shape = gt_box[:, 3:] - gt_box[:, :3]
    gt_volume_shape[gt_volume_shape < 0] = 0
    gt_volume = gt_volume_shape[:, 0] * gt_volume_shape[:, 1] * gt_volume_shape[:, 2]

    union_volume = pred_volume + gt_volume - inter_volume

    iou = inter_volume / union_volume
    return iou


def metric(pred_landmarks, spacing, landmarks):
    # [batchsize, n_class, 3]
    N = pred_landmarks.shape[0]
    n_class = pred_landmarks.shape[1]
    mre = []
    pred_num = []
    target_num = []
    hits = np.zeros((8, n_class))

    for j in range(N):
        for i in range(n_class):
            predict_landmark = pred_landmarks[j, i]

            cur_mre = np.linalg.norm(
                np.array(landmarks[j, i] - predict_landmark)*spacing[j], ord=2)

            if np.max(predict_landmark)>0:
                pred_num.append(1)
            else:
                pred_num.append(0)
            if np.mean(landmarks[j, i])>0:
                target_num.append(1)
            else:
                target_num.append(0)
                
            if np.mean(landmarks[j, i])>0:
                mre.append(cur_mre)
                hits[4:, i] += 1
                if cur_mre <= 2.0:
                    hits[0, i] += 1
                if cur_mre <= 2.5:
                    hits[1, i] += 1
                if cur_mre <= 3.:
                    hits[2, i] += 1
                if cur_mre <= 4.:
                    hits[3, i] += 1
            else:
                mre.append(-1)
            # print("[C{:2}] gt: {}, pred: {}, spacing: {}, mre(mm): {}".format(i, landmarks[j, i], predict_landmark, spacing[j], cur_mre))
    mre = np.stack(mre)
    pred_num = np.stack(pred_num)
    target_num = np.stack(target_num)

    return mre, pred_num, target_num, hits


def classify_metric(pred, target):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for pred_, target_ in zip(pred, target):
        if target_ == 1:
            if pred_ == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred_ == 1:
                FP += 1
            else:
                TN += 1
                
    return TP/(TP+FP+0.000001), 2*TP/(2*TP+FN+FP+0.000001)


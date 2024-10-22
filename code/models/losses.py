from traceback import print_tb
import torch
import torch.nn.functional as F
import torch.nn as nn
import pdb
import numpy as np


class AlignLoss(nn.Module):
    def __init__(self):
        super(AlignLoss, self).__init__()

    def forward(self, x, gt_x):
        # [Input]
        # x: [batch_size, 6]
        # gt_x: [batch_size, 6]
        #
        # [Output]
        # iou: [batch_size]
        batch_size = x.shape[0]
        loss = torch.sum(torch.sqrt(torch.sum((x - gt_x) ** 2, 1)))
        return loss / batch_size


class HNM_heatmap(nn.Module):
    def __init__(self, R=20):
        super(HNM_heatmap, self).__init__()
        self.R = R
        self.regressionLoss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, heatmap, target_heatmap):
        loss = 0
        batch_size = heatmap.size(0)
        n_class = heatmap.size(1)
        heatmap = heatmap.reshape(batch_size, n_class, -1)
        target_heatmap = target_heatmap.reshape(batch_size, n_class, -1)
        for i in range(batch_size):
            for j in range(n_class):
                select_number = torch.sum(
                    target_heatmap[i, j] >= 0).int().item()
                if select_number <= 0:
                    select_number = int(self.R * self.R * self.R / 8)
                else:
                    _, cur_idx = torch.topk(
                        target_heatmap[i, j], select_number)
                    predict_pos = heatmap[i, j].index_select(0, cur_idx)
                    target_pos = target_heatmap[i, j].index_select(0, cur_idx)
                    loss += self.regressionLoss(predict_pos, target_pos)

                mask_neg = 1 - target_heatmap[i, j]
                neg_number = torch.sum(
                    target_heatmap[i, j] < 0).int().item()
                _, neg_idx = torch.topk(mask_neg, neg_number)
                predict_neg = heatmap[i, j].index_select(0, neg_idx)
                _, cur_idx = torch.topk(predict_neg,
                                        select_number)
                predict_neg = heatmap[i, j].index_select(0, cur_idx)
                target_neg = target_heatmap[i, j].index_select(0, cur_idx)
                loss += self.regressionLoss(predict_neg, target_neg)
        return loss / batch_size

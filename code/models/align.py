import torch.nn as nn
import torch.nn.functional as F
import math

class AlignNet(nn.Module):
    def __init__(self, fc_inputsize):
        super(AlignNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=5, stride=1, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64 * fc_inputsize, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(256, 6)
        self.sigmoid4 = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = out.view(out.shape[0], -1)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.sigmoid4(out)

        return out



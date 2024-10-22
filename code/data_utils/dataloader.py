import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import pdb
import time
import nrrd

        
class Maxillofacial3D(Dataset):
    def __init__(self, transform=None, phase='train', parent_path=None, data_path="20230713CT200", data_type="all"):
        
        self.data_files = []
        self.label_files = []
        self.spacing = []
        
        with open(os.path.join(parent_path, phase+".txt"), 'r', encoding='utf-8') as f:
            ct_names = f.readlines()
        for name in ct_names:
            if data_type == "full":
                _label = np.load(os.path.join(parent_path, data_path, name[:-1]+"_label.npy"))
                if np.any(np.sum(_label,1)<0):
                    continue
            if data_type == "mini":
                _label = np.load(os.path.join(parent_path, data_path, name[:-1]+"_label.npy"))
                if np.all(np.sum(_label,1)>0):
                    continue
            self.data_files.append(os.path.join(parent_path, data_path, name[:-1]+"_volume.npy"))
            self.label_files.append(os.path.join(parent_path, data_path, name[:-1]+"_label.npy"))
            self.spacing.append(os.path.join(parent_path, data_path, name[:-1]+"_spacing.npy"))
        
        self.transform = transform
        print('the data length is %d, for %s' % (len(self.data_files), phase))

    def __len__(self):
        L = len(self.data_files)
        return L

    def __getitem__(self, index):
        _img = np.load(self.data_files[index]).astype(np.float32)
        _landmark = np.load(self.label_files[index]).astype(np.float32)
        _spacing = np.load(self.spacing[index]).astype(np.float32)
        _oimage_shape = np.array(_img.shape)
        _filename = self.data_files[index]
        
        sample = {'image': _img, 'landmark': _landmark, 'spacing':_spacing, 'oimage_shape': _oimage_shape, 'filename': _filename, 'lm': _landmark.copy()}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
        
    def __str__(self):
        pass

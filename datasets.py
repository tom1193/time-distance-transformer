import os
import numpy as np
import random
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from monai.transforms import SpatialPad
import sys
import warnings
from scipy.ndimage import zoom
sys.path.append("/home/local/VANDERBILT/litz/github/MASILab/DSB2017")
from layers import nms


def parse_str_list(str_list):
    l = re.split('[][, ]', str_list)
    try:
        return torch.tensor([int(i) for i in l if len(i) != 0], dtype=torch.int)
    except:
        return torch.tensor([float(i) for i in l if len(i) != 0], dtype=torch.float32)


class TumorCifarDataset(Dataset):
    def __init__(self, root_dir, img_names, label_df, time_range, img_transform=None, td_transform=None):
        self.root_dir = root_dir
        self.img_names = img_names
        self.label_df = label_df
        self.time_length = time_range[1] + 1  # i.e (0,2) gets T0, T1, T2. Assumes we always start at T0
        self.img_transform = img_transform
        self.td_transform = td_transform
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # returns Tx3xHxW where T is length of time range
        img_name = self.img_names[index]

        # get image data
        seq = torch.zeros((self.time_length, 3, 32, 32), dtype=torch.float32)
        for t in range(self.time_length):
            tmp = Image.open(os.path.join(self.root_dir, f"T{t}", img_name))
            seq[t] = self.to_tensor(tmp)
        if self.img_transform:
            seq = self.img_transform(seq)

        # get label
        row = self.label_df[self.label_df['img'] == img_name]
        label = int(row['gt'])

        # get time distances
        td = str(row['time'].values[0])
        times = parse_str_list(td)
        times = times[:self.time_length]
        if self.td_transform:
            times = self.td_transform(times)

        return {"name": img_name, "img_seq": seq, "label": label, "times": times}

    def __len__(self):
        return len(self.img_names)

class NLSTDataset(Dataset):
    """[PID]time[SessionID]"""
    def __init__(self, root_dir, pids, label_df, time_range=(0,1), img_transform=None):
        self.root_dir = root_dir
        self.pids = pids
        self.label_df = label_df
        self.time_length = time_range[1] + 1
        self.img_transform = img_transform

    def __getitem__(self, index):
        # returns (t c x y z)
        pid = self.pids[index]
        # returns descending in scan date
        pid_rows = self.label_df[self.label_df['PID']==pid].sort_values(by=['Session'], ascending=False)
        
        # get image data, all resized to 256, 256, 256
        seq = torch.zeros((self.time_length, 1, 256, 256, 256), dtype=torch.float32)
        fnames = pid_rows['filename']
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.root_dir, fname))
            if self.img_transform:
                tmp = self.img_transform(tmp)
            seq[t] = tmp
        
        # relative time distances in descending order
        times = pid_rows['Duration'].tolist()
        times = [t-times[0] for t in times] # set latest scan t=0
        times = torch.Tensor(times)/30.3 # transform into fractional months
        
        # get label
        label = int(pid_rows.iloc[0]['Cancer'])
        
        return {"names": pid_rows['filename'].tolist(), "img_seq": seq, "label": label, "times": times}
    
    def __len__(self):
        return len(self.pids)

class NLSTDatasetFromROI(Dataset):
    """Extract topk ROIs from Liao model for latest scan, and apply it to all the scans before"""
    def __init__(self, root_dir, pids, label_df, pbb_dir=None, patch_size=128, time_range=(0,1), topk=5, img_transform=None, phase="train"):
        self.root_dir = root_dir
        self.pids = pids
        self.label_df = label_df
        self.time_length = time_range[1] + 1
        self.img_transform = img_transform
        self.pbb_dir = pbb_dir
        self.patch_size = patch_size
        self.radius = int(patch_size/2)
        self.topk = topk
        self.crop_size = (patch_size, patch_size, patch_size)
        self.crop = simpleCrop(self.crop_size, phase=phase)

    def __getitem__(self, index):
        # returns (t c x y z)
        pid = self.pids[index]
        # returns descending in scan date
        pid_rows = self.label_df[self.label_df['PID']==pid].sort_values(by=['Session'], ascending=False)
        
        # get image data
        seq = torch.zeros((self.time_length, self.topk, self.patch_size, self.patch_size, self.patch_size), dtype=torch.float32)
        fnames = pid_rows['filename'].tolist()
        pbb = None
        for t, fname in enumerate(fnames):
            tmp = np.load(os.path.join(self.root_dir, fname))
            
            # find ROI of the latest scan and extract same region from all previous
            if t==0:
                pbb = np.load(os.path.join(self.pbb_dir, fname))
                pbb = pbb[pbb[:,0]>-1]
                pbb = nms(pbb,0.05)
            
            # crop ROIs and if less than topk ROIs, leave rest of patches as zero
            conf_list = pbb[:,0]
            chosenid = conf_list.argsort()[::-1][:self.topk]
            croplist = np.zeros([self.topk,self.crop_size[0],self.crop_size[1],self.crop_size[2]]).astype('float32')
            for k, cid in enumerate(chosenid):
                target=pbb[cid, 1:]
                crop = self.crop(tmp, target)
                crop = crop.astype(np.float32)
                if self.img_transform:
                    crop = self.img_transform(crop)
                croplist[k] = crop
            seq[t] = torch.from_numpy(croplist)
            
        # relative time distances in descending order
        times = pid_rows['Duration'].tolist()
        times = [t-times[0] for t in times] # set latest scan t=0
        times = torch.Tensor(times)/356 # transform into fractional years
        # times = torch.Tensor(times)/30.33 # transform into fractional months
        
        # get label
        label = int(pid_rows.iloc[0]['Cancer'])
        
        return {"names": pid_rows['filename'].tolist(), "img_seq": seq, "label": label, "times": times}
    
    def __len__(self):
        return len(self.pids)

class simpleCrop():
    """Cropping algorithm from Liao https://github.com/lfz/DSB2017/blob/master/data_classifier.py#L110"""
    
    def __init__(self,crop_size, scaleLim=[0.85,1.15], radiusLim=[6,100], stride=4, jitter_range=0.15, filling_value=160, phase='train'):
        self.crop_size = crop_size
        self.scaleLim = scaleLim
        self.radiusLim = radiusLim
        self.stride = stride
        self.jitter_range = jitter_range
        self.filling_value = filling_value
        self.phase = phase
        
    def __call__(self,imgs,target):
        crop_size = np.array(self.crop_size).astype('int')
        if self.phase=='train':
            jitter_range = target[3]*self.jitter_range
            jitter = (np.random.rand(3)-0.5)*jitter_range
        else:
            jitter = 0
        start = (target[:3]- crop_size/2 + jitter).astype('int')
        pad = [[0,0]]
        for i in range(3):
            if start[i]<0:
                leftpad = -start[i]
                start[i] = 0
            else:
                leftpad = 0
            if start[i]+crop_size[i]>imgs.shape[i+1]:
                rightpad = start[i]+crop_size[i]-imgs.shape[i+1]
            else:
                rightpad = 0
            pad.append([leftpad,rightpad])
        imgs = np.pad(imgs,pad,'constant',constant_values =self.filling_value)
        crop = imgs[:,start[0]:start[0]+crop_size[0],start[1]:start[1]+crop_size[1],start[2]:start[2]+crop_size[2]]

        return crop

class NLSTDatasetFromFeat(Dataset):
    def __init__(self, root_dir, pids, label_df, feat_dim=64, time_range=(0,1), topk=5, img_transform=None):
        self.root_dir = root_dir
        self.pids = pids
        self.feat_dim = feat_dim
        self.label_df = label_df
        self.time_length = time_range[1] + 1
        self.topk = topk
        self.img_transform = img_transform

    def __getitem__(self, index):
        # returns (t c x y z)
        pid = self.pids[index]
        # returns descending in scan date
        pid_rows = self.label_df[self.label_df['PID']==pid].sort_values(by=['Session'], ascending=False)
        
        # get feature vectors
        seq = np.zeros((self.time_length, 5, self.feat_dim), dtype='float32')
        fnames = pid_rows['filename'].apply(lambda x: x.split(".")[0]).tolist()
        for t, fname in enumerate(fnames):
            feat = np.load(os.path.join(self.root_dir, f"{fname}.npy"))[:self.topk]
            seq[t] = self.img_transform(feat)
        
        # relative time distances in descending order
        times = pid_rows['Duration'].tolist()
        times = [t-times[0] for t in times] # set latest scan t=0
        times = torch.Tensor(times)/30.33 # transform into fractional months
        
        # get label
        label = int(pid_rows.iloc[0]['Cancer'])
        
        return {"names": pid_rows['filename'].tolist(), "img_seq": seq, "label": label, "times": times}
    
    def __len__(self):
        return len(self.pids)

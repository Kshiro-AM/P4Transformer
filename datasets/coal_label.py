import os
import sys
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

class CoalLabelDataset(Dataset):
    def __init__(self, root='data/coal', meta='data/train_raw.txt', frames_per_clip=3, num_points=16384, train=False):
        super(CoalLabelDataset, self).__init__()
        
        self.num_points = num_points
        self.train = train
        self.root = root
        self.frames_per_clip = frames_per_clip
        
        self.labelweights = np.ones(2, dtype=np.float32)
        
        self.meta = []
        self.data = np.load(os.path.join(root, 'coal_labeled.npz'))
        
        self.label = self.data['label']
        self.pc = self.data['pc']
        self.pre = self.data['pre']
        
        
    def __getitem__(self, index):

        # rd_idx1 = np.random.randint(0, self.pc.shape[0])
        # rd_idx2 = np.random.randint(0, self.pc.shape[0])
        # clip = np.dstack((self.pc[index], self.pc[rd_idx1], self.pc[rd_idx2]))
        clip = np.dstack((self.pc[index], self.pc[index+1], self.pc[index+2]))
        clip = np.swapaxes(clip, 1, 2)
        clip = np.swapaxes(clip, 0, 1)
        
        # label = np.dstack((self.label[index], self.label[rd_idx1], self.label[rd_idx2]))
        label = np.dstack((self.label[index], self.label[index+1], self.label[index+2]))
        label = label.reshape((label.shape[1], label.shape[2]))
        label = np.swapaxes(label, 0, 1)

        rgb = np.zeros_like(clip)
        rgb = np.swapaxes(rgb, 1, 2)
        
        pre = np.dstack((self.pre[index], self.pre[index+1], self.pre[index+2]))

        return clip.astype(np.float32), rgb.astype(np.float32), label.astype(np.int64), pre.astype(np.int64), index
    
    def __len__(self):
        return self.pc.shape[0] - 2
    
    
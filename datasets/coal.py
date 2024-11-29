import os
import sys
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset

class CoalDataset(Dataset):
    def __init__(self, root='data/coal', meta='data/train_raw.txt', frames_per_clip=3, num_points=16384, train=False):
        super(CoalDataset, self).__init__()
        
        # just for test now
        
        self.num_points = num_points
        self.train = train
        self.root = root
        self.frames_per_clip = frames_per_clip
        
        self.labelweights = np.ones(2, dtype=np.float32)
        
        self.meta = []
        self.data = np.load(os.path.join(root, 'coal.npz'))
        self.pc = self.data['pc']
        self.pre = self.data['pre']
        
    def __getitem__(self, index):

        clip = np.dstack((self.pc[index], self.pc[index+1], self.pc[index+2]))
        clip = np.swapaxes(clip, 1, 2)
        clip = np.swapaxes(clip, 0, 1)

        rgb = np.zeros_like(clip)
        rgb = np.swapaxes(rgb, 1, 2)
        
        pre = np.dstack((self.pre[index], self.pre[index+1], self.pre[index+2]))

        return clip.astype(np.float32), rgb.astype(np.float32), pre.astype(np.int64), index
    
    def __len__(self):
        return self.data['pc'].shape[0] - 2
    
    
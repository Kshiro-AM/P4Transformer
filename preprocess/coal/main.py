import os
import numpy as np
from enum import Enum
from sklearn.neighbors import KDTree

import open3d as o3d

class DisMeasure:
    def __init__(self, sourceClouds):
        self.sourceClouds = sourceClouds

    def match_by_hausdorffdis(self, inCloud):
        
        tree = KDTree(inCloud)
        
        dis = []
        for i, sourceCloud in enumerate(self.sourceClouds):
            hausdis = 0
            dists, _ = tree.query(sourceCloud, k=1)
            hausdis = max(dists)
            dis.append(hausdis)
            
        return np.array(dis).squeeze()

class DatasetType(Enum):
    train = 0
    test = 1

def get_all_files(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def read_pcd_binary(pcd_file):

    with open(pcd_file, 'rb') as f:
        # pcd header
        pcd = o3d.io.read_point_cloud(pcd_file, format='pcd', remove_nan_points=True,
                                  remove_infinite_points=True, print_progress=True)
    
    return pcd

def read_ply_ascii(ply_file):
    
    with open(ply_file, 'r') as f:
        header = ""
        propertyDict = {}
        propertyCount = 0
        while True:
            line = f.readline()
            header += line
            if line.startswith("property"):
                propertyDict[line.split()[2]] = propertyCount
                propertyCount += 1
            if line.startswith("end_header"):
                break
        
        pc = []
        label = []
        while True:
            line = f.readline()
            if not line:
                break
            p = line.split()
            pc.append([ float(p[propertyDict['x']]), float(p[propertyDict['y']]), float(p[propertyDict['z']]) ])
            label.append(int(float(p[propertyDict['scalar_Original_cloud_index']])))
            
        return header, np.array(pc, dtype=np.float32), np.array(label, dtype=np.int32)

def main():
    dir = '../../data/lidar_pcd'
    file_list = get_all_files(dir)
    file_list.sort()
    
    point_size = 16384
    
    templates = np.load(os.path.join('../../data/coal/', 'coal_template.npz'))['pc']
    disMeasure = DisMeasure(templates)
    
    pc = np.zeros(shape=[len(file_list), point_size, 3]) # frame, point size, xyz
    pre = np.zeros(shape=[len(file_list), len(templates)], dtype=np.float32) # frame, template num
    label = np.zeros(shape=[len(file_list), point_size], dtype=np.int32) # frame, point size
    for i, file in enumerate(file_list):
        print('Processing: ', file)
        cloud = read_pcd_binary(file)
        if np.asarray(cloud.points).shape[0] > point_size:
            cloud = cloud.farthest_point_down_sample(point_size)
        cloud = np.asarray(cloud.points)
            
        for j in range(cloud.shape[0]):
            try:
                pc[i][j][0] = cloud[j][0]
                pc[i][j][1] = cloud[j][1]
                pc[i][j][2] = cloud[j][2]
            except:
                print('error')
                
        pre[i] = disMeasure.match_by_hausdorffdis(pc[i])
          
            
    np.savez('../../data/coal/coal.npz', pc=pc, label=label, pre=pre)
    
 
if __name__ == '__main__':
    main()

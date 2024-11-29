import os
import numpy as np
from enum import Enum

import open3d as o3d

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
        while True:
            line = f.readline()
            if not line:
                break
            p = line.split()
            pc.append([ float(p[propertyDict['x']]), float(p[propertyDict['y']]), float(p[propertyDict['z']]) ])
            
        return header, np.array(pc, dtype=np.float32)

def main():
    dir = '../../data/coal_template'
    file_list = get_all_files(dir)
    file_list.sort()
    
    point_size = 2048
    
    pc = np.zeros(shape=[len(file_list), point_size, 3]) # frame, point size, xyz
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
                print('ERROR')
        
            
            
    np.savez('../../data/coal/coal_template.npz', pc=pc)
    
 
if __name__ == '__main__':
    main()

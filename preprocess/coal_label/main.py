import os
import numpy as np
import open3d as o3d
import torch

from enum import Enum

def save_ply(points, colors, filename, pred):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % points.shape[0])
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property int class\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            f.write('%f %f %f %d %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], colors[i][0], colors[i][1], colors[i][2], pred[i]))

def farthest_point_sampling(arr, n_sample, start_idx=None):
    """Farthest Point Sampling without the need to compute all pairs of distance.

    Parameters
    ----------
    arr : numpy array
        The positional array of shape (n_points, n_dim)
    n_sample : int
        The number of points to sample.
    start_idx : int, optional
        If given, appoint the index of the starting point,
        otherwise randomly select a point as the start point.
        (default: None)

    Returns
    -------
    numpy array of shape (n_sample,)
        The sampled indices.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(100, 1024)
    >>> point_idx = farthest_point_sampling(data, 3)
    >>> print(point_idx)
        array([80, 79, 27])

    >>> point_idx = farthest_point_sampling(data, 5, 60)
    >>> print(point_idx)
        array([60, 39, 59, 21, 73])
    """
    n_points, n_dim = arr.shape

    if (start_idx is None) or (start_idx < 0):
        start_idx = np.random.randint(0, n_points)

    sampled_indices = [start_idx]
    min_distances = np.full(n_points, np.inf)
    
    for _ in range(n_sample - 1):
        current_point = arr[sampled_indices[-1]]
        dist_to_current_point = np.linalg.norm(arr - current_point, axis=1)
        min_distances = np.minimum(min_distances, dist_to_current_point)
        farthest_point_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_point_idx)

    return np.array(sampled_indices)

class DatasetType(Enum):
    train = 0
    test = 1

def get_all_files(dir_path):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def read_ply_ascii(ply_file):
    device = o3d.core.Device("CPU:0")
    with open(ply_file, 'r') as f:
        tPointCloud = o3d.t.geometry.PointCloud(device)
        ply = o3d.io.read_point_cloud(ply_file, format='ply', remove_nan_points=True,
                                  remove_infinite_points=True, print_progress=True, )
        pointcloud = np.asarray(ply.points)
        tPointCloud.point.positions = o3d.core.Tensor(pointcloud)
        ply = o3d.t.geometry.PointCloud(tPointCloud)
        
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
        
        label = []
        while True:
            line = f.readline()
            if not line:
                break
            p = line.split()
            label.append(int(float(p[propertyDict['scalar_Original_cloud_index']])))
            
        ply.point.labels = o3d.core.Tensor(label)
            
        return header, ply

def main():
    dir = '../../data/ply'
    file_list = get_all_files(dir)
    file_list.sort()
    
    point_size = 16384
    
    pc = np.zeros(shape=[len(file_list), point_size, 3]) # frame, point size, xyz
    label = np.zeros(shape=[len(file_list), point_size], dtype=np.int32) # frame, point size
    for i, file in enumerate(file_list):
        print('Processing: ', file)
        header, cloud = read_ply_ascii(file)
        
        # with downsample
        length = len(cloud.point.positions)
        if np.asarray(cloud.point.positions).shape[0] > point_size:
            arr = cloud.point.positions.cpu().numpy()
            idxs = farthest_point_sampling(arr, point_size)
            length = point_size
        else:
            idxs = np.arange(length)
            
        true_index = []
        for j, l in enumerate(cloud.point.labels):
            if l.cpu().numpy() == 0:
                true_index.append(j)
        
        labels = cloud.point.labels.cpu().numpy()
        true_labels = labels[true_index]
        
        points = cloud.point.positions.cpu().numpy()
        true_points = points[true_index]
            
        for j in range(length):
            pc[i, j] = points[idxs[j]]
            label[i, j] = labels[idxs[j]]
            
            sort_idx = np.argsort(label[i])
            label[i] = label[i][sort_idx]
            pc[i] = pc[i][sort_idx]
            
        # save_ply(points[idxs], np.zeros_like(points[idxs]), file.replace('.ply', '_downsampled.ply').replace('/ply/', '/temp/'), labels[idxs])
            
    np.savez('../../data/coal/coal_labeled.npz', pc=pc, label=label)
    
 
if __name__ == '__main__':
    main()

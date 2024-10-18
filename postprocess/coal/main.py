import os
import numpy as np

def get_all_files(dir_path, suffix=None):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if suffix and suffix == file.split('.')[-1]:
                file_list.append(os.path.join(root, file))
    return file_list

def read_ply(file):
    points = []
    preds = []
    content = False
    with open(file) as f:
        line = f.readline()
        while line:
            if content:
                list = line.split(' ')
                point = [float(list[0]), float(list[1]), float(list[2])]
                points.append(point)
                preds.append(int(list[-1]))
            if not content and line == 'end_header\n':
                content = True
            line = f.readline()
            
    return points, preds
            
def main():
    output_dir = '/home/aeolus/dev/P4Transformer/data/coal_result'
    
    # bonding box
    min_x = 0.2
    min_y = 0.9
    max_y = 1.2
    min_z = -0.3
    max_z = 0.0
    
    file_list = get_all_files(output_dir, 'ply')
    
    max = -99.0
    min = 99.0
    for file_name in file_list:
        points, preds = read_ply(os.path.join(output_dir, file_name))
        for i, point in enumerate(points):
            if preds[i] == 5 and point[0] > min_x and point[1] > min_y and point[1] < max_y and point[2] > min_z and point[2] < max_z:
                if point[0] < min:
                    min = point[0]
                if point[0] > max:
                    max = point[0]
                    
    print('max: %f\nmin: %f' % (max, min))

if __name__ == '__main__':
    main()
""" read points data file

Author: Henry Zhang
Date:January 25, 2020
"""

# module
from point_cloud import PointCloud
from plane_3d import Plane3D
import numpy as np

# parameters


# classes


# functions
def read_points_raw(filename):
    data = np.loadtxt(filename, delimiter=',')
    pcd = PointCloud(data[:,1:4].T)             
    return pcd

def read_points_data(filename):
    planes = []
    point_clouds =  []
    with open(filename, "r") as fp:
        lines = fp.readlines()
        if lines[0][0] == 'T':
            for line in lines[1:4]:
                plane_param = list(map(float, line.split(",")))
                planes.append(Plane3D.create_plane_from_list(plane_param))
            last_num = 0
            start_line = 6
            end_line = start_line
            for line in lines[6::]:
                if line[0] != '\n' and end_line != len(lines)-1 and int(line[0]) == last_num:
                    end_line += 1
                else:
                    end_line += 1
                    data = np.zeros((3, end_line-start_line))
                    if line[0] == '\n' :
                        break
                    for idx, line in enumerate(lines[start_line:end_line]):
                        points = list(map(float, line.split(',')))
                        data[:, idx] = points[1::]
                    point_clouds.append(PointCloud(data))
                    start_line = end_line
                    last_num = int(line[0])
                    
    return planes, point_clouds

def test_read_points_data():
    filename = "/home/henry/Documents/projects/avl/Detection/ground/data/points_data_1.txt"
    planes, point_clouds = read_points_data(filename)
    print(planes)

def test_read_points_raw():
    filename = "/home/henry/Documents/projects/avl/Detection/ground/data/points_raw_2.txt"
    pcd = read_points_raw(filename)
    print(pcd.data.shape)

# main
def main():
    # test_read_points_data()
    test_read_points_raw()

if __name__ == "__main__":
    main()
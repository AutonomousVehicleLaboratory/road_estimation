""" estimate road parameter from point cloud

Author: Henry Zhang
Date:January 25, 2020
"""

# module
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from read_file import read_points_data
from plane_3d import Plane3D
from rotation import Rotation
from ransac import RANSAC
from point_cloud import PointCloud

# parameters


# classes


# functions
def clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold):
    plane = Plane3D.create_plane_from_vectors_and_point(vec1, vec2, pt1)
    distance = plane.distance_to_plane(pcd.data.T)
    data_close = pcd.data[:,distance<threshold]
    data_far = pcd.data[:,distance>=threshold]
    pcd_close = PointCloud(data_close)
    pcd_far = PointCloud(data_far)
    return pcd_close, pcd_far


def vis(plane, pcd, dim_2d=True):
    fig = plt.figure(figsize=(16,8))
    if dim_2d == True:
        ax = fig.add_subplot(211)
        pcd.color_points_by_distance_plane(plane, threshold=0.1)
        plot = pcd.vis(ax, dim_2d=True, s=3, lim=[-20, 40, -12, 12])
        plt.colorbar(plot)
        ax2 = fig.add_subplot(212)
        pcd.color_points_by_distance_plane(plane, threshold=0.0, clip = 0.5)
        plot = pcd.vis(ax2, dim_2d=True, lim=[-20, 40, -12, 12])
        plt.colorbar(plot)
    else:
        ax = fig.add_subplot(111, projection='3d')
        plane.vis(ax)
        plot = pcd.vis(ax, dim_2d=False)
        plt.colorbar(plot)
    


# main
def main():
    filenum = 1
    lidar_pitch = 9.5/180*np.pi
    
    filename = "/home/henry/Documents/projects/avl/Detection/ground/data/points_data_"+repr(filenum)+".txt"
    planes, pcds = read_points_data(filename)
    
    tf = Rotation(roll=0.0, pitch=lidar_pitch, yaw=0.0)
    for plane in planes:
        plane.rotate_around_axis(axis="y", angle=-lidar_pitch)
    for pcd in pcds:
        pcd.rotate(tf)
    
    
    pcd = pcds[0]
    plane = planes[1]
    # print(np.min(pcd.data[0,:]), np.max(pcd.data[0,:]))
    vis(plane, pcd, dim_2d=True)

    # the onlyfloor points
    # pcd = pcds[2]
    # plane = planes[2]
    # the reestimated plane
    sac = RANSAC(Plane3D, min_sample_num=3, threshold=0.1, iteration=50, method="MSAC")
    plane2, _, _, _ = sac.ransac(pcd.data.T)
    vis(plane2, pcd, dim_2d=True)
    plt.show()

if __name__ == "__main__":
    main()
""" Utility functions

Author: Henry Zhang
Date:February 12, 2020
"""

# module
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from plane_3d import Plane3D
from point_cloud import PointCloud

# parameters


# classes


# functions
def clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold):
    """ given planes specified by two vectors and a point, threshold the point cloud
    by signed distance

    Param:
        pcd: PointCloud type
        vec1, vec2, pt1: 3*1 arrays
        threshold: (2,) list gives the [max, min] of signed distance to the plane
    Return:
        pcd_close, pcd_far: separated point cloud."""
    plane = Plane3D.create_plane_from_vectors_and_point(vec1, vec2, pt1)
    distance = plane.distance_to_plane_signed(pcd.data.T)
    idx_close =  np.logical_and(distance<threshold[0], distance>threshold[1])
    idx_far = np.logical_or(distance>=threshold[0], distance<=threshold[1])
    data_close = pcd.data[:,idx_close]
    data_far = pcd.data[:,idx_far]
    pcd_close = PointCloud(data_close)
    pcd_far = PointCloud(data_far)
    return pcd_close, pcd_far

def test_clip_pcd_by_distance_plane(pcd):
    vec1 = np.array([1,0,0])
    vec2 = np.array([0,0,1])
    pt1 = np.array([0,0,0])
    threshold = [6.0, -3]
    pcd_close, _ = clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    pcd_close.vis(ax)
    # plt.show()

def homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))

def dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1]/x[-1]

# main
def main():
    pass

if __name__ == "__main__":
    main()
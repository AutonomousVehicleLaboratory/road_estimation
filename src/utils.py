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
# def clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold):
#     """ given planes specified by two vectors and a point, threshold the point cloud
#     by signed distance

#     Param:
#         pcd: PointCloud type
#         vec1, vec2, pt1: 3*1 arrays
#         threshold: (2,) list gives the [max, min] of signed distance to the plane
#     Return:
#         pcd_close, pcd_far: separated point cloud."""
    
#     distance = plane.distance_to_plane_signed(pcd.data.T)
#     idx_close =  np.logical_and(distance<threshold[0], distance>threshold[1])
#     idx_far = np.logical_or(distance>=threshold[0], distance<=threshold[1])
#     data_close = pcd.data[:,idx_close]
#     data_far = pcd.data[:,idx_far]
#     pcd_close = PointCloud(data_close)
#     pcd_far = PointCloud(data_far)
#     return pcd_close, pcd_far

def clip_pcd_by_distance_plane(pcd, plane, threshold):
    """ given planes specified by two vectors and a point, threshold the point cloud
    by signed distance

    Param:
        pcd: PointCloud type
        plane: Plane3D type
        threshold: (2,) list gives the [max, min] of signed distance to the plane
    Return:
        pcd_close, pcd_far: separated point cloud."""
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
    plane = Plane3D.create_plane_from_vectors_and_point(vec1, vec2, pt1)
    pcd_close, _ = clip_pcd_by_distance_plane(pcd, plane, threshold)
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

from scipy.linalg import logm, expm

# Note that np.sinc is different than defined in class
def sinc(x):
    # Returns a scalar valued sinc value
    """your code here"""
    if x == 0:
        y = 1
    else:
        y = np.sin(x) / x
    
    return y

def differentiate_sinc(x):
    if x == 0:
        return 0
    else:
        return np.cos(x)/x - np.sin(x)/(x**2)

def skew(w):
    # Returns the skew-symmetrix represenation of a vector
    """your code here"""
    w = w.reshape([3,1])
    w_skew = np.array([[     0., -w[2,0],  w[1,0]],
                       [ w[2,0],      0., -w[0,0]],
                       [-w[1,0],  w[0,0],      0.]])
    
    return w_skew

def de_skew(w_skew):
    w = np.array([[-w_skew[1,2], w_skew[0,2], -w_skew[0,1]]]).T
    return w

def singularity_normalization(w):
    """ w has a singularity at 2 pi, check every time change w """
    theta = np.linalg.norm(w)
    if theta > np.pi:
        w = (1 - 2*np.pi/ theta*np.ceil((theta - np.pi) / (2*np.pi) )) * w
    return w

def parameterize_rotation(R):
    # Parameterizes rotation matrix into its axis-angle representation
    """your code here"""
    # lecture implementation
    U, D, VT = np.linalg.svd(R - np.eye(R.shape[0]))
    v = VT.T[:,-1::]
    v_hat = np.array([[R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]]).T
    theta_sin = np.matmul(v.T , v_hat) / 2.
    theta_cos = (np.trace(R) - 1.) / 2.
    theta = np.arctan2(theta_sin, theta_cos).item()
    w = theta * v / np.linalg.norm(v)
    
    # scipy implementation
    # w_skew_2 = logm(R)
    # w_2 = DeSkew(w_skew_2)
    # theta_2 = np.linalg.norm(w_2)
    w = singularity_normalization(w)
    theta = np.linalg.norm(w)
    
    if theta < 1e-7:
        w = v_hat / 2.0
    
    theta = np.linalg.norm(w)
    return w, theta

def deparameterize_rotation(w):
    # Deparameterizes to get rotation matrix
    """your code here"""
    w = w.reshape([3,1])
    w_skew = skew(w)
    theta = np.linalg.norm(w)
    if theta < 1e-7:
        R = np.eye(w.shape[0]) + w_skew
    else:
        R = np.cos(theta) * np.eye(w.shape[0]) + sinc(theta) * w_skew + (1-np.cos(theta)) / theta**2 * np.matmul(w, w.T)
    
    return R

def jacobian_vector_norm(v):
    assert(v.shape[1] == 1)
    J = 1. / np.linalg.norm(v) * v.T
    return J

# main
def main():
    pass

if __name__ == "__main__":
    main()
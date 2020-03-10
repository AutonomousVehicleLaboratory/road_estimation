""" Utility functions

Author: Henry Zhang
Date:February 12, 2020
"""

# module
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

# parameters


# classes


# functions

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

def cross(v1, v2):
    assert(v1.shape == (3,))
    assert(v2.shape == (3,))
    return np.array([v1[1]*v2[2]-v2[1]*v1[2], v1[2]*v2[0]-v2[2]*v1[0], v1[0]*v2[1]-v2[0]*v1[1]])

# main
def main():
    pass

if __name__ == "__main__":
    main()
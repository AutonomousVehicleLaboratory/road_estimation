""" Point cloud class

Author: Henry Zhang
Date:January 25, 2020
"""

# module
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plane_3d import Plane3D
# import mayavi.mlab as mlab
# parameters


# classes
class PointCloud:
    def __init__(self, data, color="r"):
        self.data = data # 3 by n
        self.color = color # n by 1 
    
    def rotate(self, tf):
        self.data = np.matmul(tf.getMatrix() , self.data)
    
    def clip_by_x(self, threshold):
        idx = np.logical_and(self.data[0, :] > threshold[0], self.data[0,:] < threshold[1])
        data_in = self.data[:,idx]
        data_out = self.data[:,np.logical_not(idx)]
        return PointCloud(data_in), PointCloud(data_out)
    
    def color_points_by_distance_x(self, threshold = 0.0):
        distance = np.abs(self.data[0,:])
        if threshold == 0:
            # color based on distance (alpha)
            self.color = distance.reshape([-1]) / np.max(distance)
        else:
            # color based on threshold
            self.color = distance.reshape([-1]) / threshold
            self.color[self.color>1] = 1
            self.color[self.color<1] = 0.5

    def color_points_by_distance_plane(self, plane, threshold=0.0, clip = 0.0, signed = False, lists = [], RGB=False):
        if signed == True and len(lists) != 0:
            distance = plane.distance_to_plane_signed(self.data.T)
            if RGB == True:
                self.color = np.zeros((distance.shape[0],3))
                for i in range(len(lists)-1):
                    idx_in = np.logical_and(distance > lists[i], distance < lists[i+1])
                    lam = i / float(len(lists)-1)
                    self.color[idx_in, :] = np.array([lam, 0, 1-lam])
            else:
                self.color = np.zeros(distance.shape)
                for i in range(len(lists)-1):
                    idx_in = np.logical_and(distance > lists[i], distance < lists[i+1])
                    self.color[idx_in] = i-3 # / float(len(lists)-2)
        else:
            distance = plane.distance_to_plane(self.data.T)
            if threshold == 0:
                # color based on distance (alpha)
                dist_max = np.max(distance)
                if clip > 0 and dist_max > clip:
                    self.color = distance / clip
                    self.color[self.color>1] = 1
                else:
                    self.color = distance / dist_max
                # self.color = np.ones((self.data.shape[1], 4))*255
                # self.color[:,3] = distance
            else:
                # color based on threshold
                self.color = distance / threshold
                self.color[self.color>1] = 1
                self.color[self.color<1] = 0.5
                # self.color = np.ones((self.data.shape[1], 4))
                # self.color[:,0:3] = 255
                # self.color[distance>threshold,0] = 100
        pass
    
    def vis(self, ax, dim_2d=True, s=1, lim=[-20, 40, -18, 18], cmap = None, side=False):
        if dim_2d == True:
            if side == True:
                displayaxis = [0, 2]
            else:
                displayaxis = [0, 1]
            if cmap is None:
                plot = ax.scatter(self.data[displayaxis[0],:], self.data[displayaxis[1],:], c=self.color, marker=".", s=s)
            else:
                plot = ax.scatter(self.data[displayaxis[0],:], self.data[displayaxis[1],:], c=self.color, cmap=cmap, marker=".", s=s)
        else:
            plot = ax.scatter(self.data[0,:], self.data[1,:], self.data[2,:], c=self.color, marker=".", s=s)
            
        # mlab.points3d(self.data[0,:], self.data[1,:], self.data[2,:], colormap="RdYlBu", scale_factor=0.02,
        #      scale_mode='none', mode='2dcross')
        ax.set_xlim(lim[0:2])
        ax.set_ylim(lim[2:4])
        return plot



# functions

def clip_pcd_by_distance_plane(pcd, plane, threshold, in_only=False):
    """ given planes specified by two vectors and a point, threshold the point cloud
    by signed distance

    Param:
        pcd: PointCloud type
        plane: Plane3D type
        threshold: (2,) list gives the [max, min] of signed distance to the plane
        in_only: default to false, return both point cloud in and out.
    Return:
        pcd_close(, pcd_far): separated point cloud, return pcd_far if in_only is False"""
    distance = plane.distance_to_plane_signed(pcd.data.T)
    idx_close =  np.logical_and(distance<threshold[0], distance>threshold[1])
    data_close = pcd.data[:,idx_close]
    pcd_close = PointCloud(data_close)
    if in_only:
        return pcd_close
    else:
        idx_far = np.logical_or(distance>=threshold[0], distance<=threshold[1])
        data_far = pcd.data[:,idx_far]
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


# main
def main():
    pass

if __name__ == "__main__":
    main()
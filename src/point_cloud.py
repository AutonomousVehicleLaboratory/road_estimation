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

    def extract_low(self, threshold = np.array([[-10, 50], [-3.0, 6.0]]), d = 1.0):
        data = self.data.T
        width = int((threshold[0,1] - threshold[0,0]) / d)
        height = int((threshold[1,1] - threshold[1,0]) / d)
        idx = ((data[:,0] - threshold[0,0]) / d).astype(np.int)
        idy = ((data[:,1] - threshold[1,0]) / d).astype(np.int)
        xin = np.logical_and(0 <= idx, idx < width)
        yin = np.logical_and(0 <= idy, idy < height)
        zin = np.logical_and(xin, yin)
        data_id = idx[zin] * height + idy[zin]
        data_val = data[zin]
        max_val = np.max(data_val) + 1
        ans = np.ones((width*height, 3))*max_val
        for i in range(data_val.shape[0]):
            j = data_id[i]
            if data_val[i,2] < ans[j,2]:
                ans[j] = data_val[i]
        id_valid = np.all(ans != max_val * np.array([[1.,1.,1.]]), axis=1)
        data_low = ans[id_valid]
        return PointCloud(data_low.T)

    
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

# def reverse_xz(data):
#     temp = np.array(data[:,2])
#     data[:,2] = data[:,0]
#     data[:,0] = temp
#     return data

# def reverse_xz_to_tuple_row(data):
#     arr = np.empty(data.shape[0], dtype=tuple)
#     arr[:] = map(tuple, data)
#     return arr

def extract_low(data, threshold = np.array([[-10, 50], [-3.0, 6.0]]), d = 1.0):
    width = int((threshold[0,1] - threshold[0,0]) / d)
    height = int((threshold[1,1] - threshold[1,0]) / d)
    idx = ((data[:,0] - threshold[0,0]) / d).astype(np.int)
    idy = ((data[:,1] - threshold[1,0]) / d).astype(np.int)
    xin = np.logical_and(0 <= idx, idx < width)
    yin = np.logical_and(0 <= idy, idy < height)
    zin = np.logical_and(xin, yin)
    data_id = idx[zin] * height + idy[zin]
    data_val = data[zin]
    # data_val = reverse_xz_to_tuple_row(data_val)
    max_val = np.max(data_val) + 1
    ans = np.ones((width*height, 3))*max_val
    for i in range(data_val.shape[0]):
        j = data_id[i]
        if data_val[i,2] < ans[j,2]:
            ans[j] = data_val[i]
    # data_id = np.array([0,0,0,1,1,1,1,2,2,2,3,3,3,4,5,5,5])
    # data_val = np.random.randint(0, 100, size=(len(data_id), 3))
    # data_val = reverse_xz(data_val)
    # ans = np.empty((data_id[-1]+1, 3)) # might want to use max(data_id) and zeros instead
    # np.minimum.at(ans,data_id,data_val)
    # ans = reverse_xz(ans)
    id_valid = np.all(ans != max_val * np.array([[1.,1.,1.]]), axis=1)
    data_low = ans[id_valid]
    return data_low

def test_extract_low():
    import pickle, pptk
    close = False
    filename = "/home/henry/Documents/projects/pylidarmot/src/road_estimation/data/pcd.pkl"
    with open(filename, 'rb') as fp:
        pcd = pickle.load(fp)
    # data = np.array([
    #     [0, 1, 2, 3, 4, 5, 6],
    #     [2, 3, 4, 3, 4, 5, 3],
    #     [0, 1, 0, 1, 0, 1, 0]
    # ]).T
    # data = np.random.rand(1000, 3)
    # data = data * np.array([[10, 9, 1]]) - np.array([[0, 3, 0]])
    # pcd = PointCloud(data.T)
    pcd, _ = pcd.clip_by_x([-10, 60])
    vec1 = np.array([1,0,0])
    vec2 = np.array([0,0,1])
    pt1 = np.array([0,0,0])
    threshold = [-3.0, 6.0]
    plane_from_vec = Plane3D.create_plane_from_vectors_and_point(vec1, vec2, pt1)
    pcd_close = clip_pcd_by_distance_plane(pcd, plane_from_vec, threshold, in_only=True)
    
    pcd_low = extract_low(pcd_close.data.T)
    # pcd_low = np.zeros(pcd_low.shape)
    vis_data = np.vstack([pcd_close.data.T, pcd_low])

    attribute = np.ones(vis_data.shape).astype(np.float)
    attribute[-pcd_low.shape[0]:-1, 1::] = 0 
    v = pptk.viewer(vis_data)
    v.attributes(attribute, vis_data[:,2]/ np.max(vis_data[:,2]))
    v.set(point_size=0.01)
    # pose = [20,10,-10,1,-np.pi/2,10]
    # pose = [40,15,-40,-np.pi/6,-np.pi/3,5]
    # v.play(pose, ts = [0], tlim=[0, 1], repeat=False)
    if close:
        time.sleep(1)
        v.close()

def clip_pcd_by_distance_plane(pcd, plane, threshold, in_only=False):
    """ given planes specified by two vectors and a point, threshold the point cloud
    by signed distance

    Param:
        pcd: PointCloud type
        plane: Plane3D type
        threshold: (2,) list gives the [min, max] of signed distance to the plane
        in_only: default to false, return both point cloud in and out.
    Return:
        pcd_close(, pcd_far): separated point cloud, return pcd_far if in_only is False"""
    distance = plane.distance_to_plane_signed(pcd.data.T)
    idx_close =  np.logical_and(distance<threshold[1], distance>threshold[0])
    data_close = pcd.data[:,idx_close]
    pcd_close = PointCloud(data_close)
    if in_only:
        return pcd_close
    else:
        idx_far = np.logical_or(distance>=threshold[1], distance<=threshold[0])
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
    test_extract_low()

if __name__ == "__main__":
    main()
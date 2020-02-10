""" Integration test

Author: Henry Zhang
Date:February 01, 2020
"""


# module
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec

from read_file import read_points_data, read_points_raw
from plane_3d import Plane3D
from rotation import Rotation
from ransac import RANSAC
from point_cloud import PointCloud
from camera import Camera
from bounding_box import BoundingBox

np.set_printoptions(precision=3)

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

def vis(plane, pcd, dim_2d=True):
    fig = plt.figure(figsize=(16,8))
    if dim_2d == True:
        ax = fig.add_subplot(211)
        pcd.color_points_by_distance_plane(plane, threshold=0.15)
        plot = pcd.vis(ax, dim_2d=True, s=3, lim=[-20, 40, -12, 12])
        plt.colorbar(plot)
        ax.set_title('[{:.2f},{:.2f}] boundary'.format(-0.15, 0.15))

        ax2 = fig.add_subplot(212)
        thresholds = [-np.inf, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, np.inf]
        pcd.color_points_by_distance_plane(plane, threshold=0.0, clip = 0.5, signed=True, lists=thresholds)
        colors = ['red','orange','yellow','green','blue','purple', 'gray'] #, 'pink', 'orange', 'cyan']
        plot = pcd.vis(ax2, dim_2d=True, lim=[-20, 40, -12, 12], cmap=clr.ListedColormap(colors))
        ax2.set_title("threshold: -0.3, -0.15, -0.05, 0.05, 0.15, 0.3")
        plt.colorbar(plot)
        
        """ 
        # failed to visualize the side view, needs an orthogonal vector, otherwise very noisy
        ax2 = fig.add_subplot(212)
        thresholds = [-np.inf, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, np.inf]
        pcd.color_points_by_distance_plane(plane, threshold=0.0, clip = 0.5, signed=True, lists=thresholds)
        colors = ['red','orange','yellow','green','blue','purple', 'gray'] #, 'pink', 'orange', 'cyan']
        plot = pcd.vis(ax2, dim_2d=True, lim=[-20, 40, -5, 0], cmap=clr.ListedColormap(colors), side=True)
        ax2.set_title("threshold: -0.3, -0.15, -0.05, 0.05, 0.15, 0.3")
        plt.colorbar(plot)
        """
    else:
        ax = fig.add_subplot(111, projection='3d')
        plane.vis(ax)
        plot = pcd.vis(ax, dim_2d=False)
        plt.colorbar(plot)

def vis_multiple(planes, pcds, dim_2d=True):
    fig = plt.figure(figsize=(16,8))
    
    if dim_2d == True:
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        title = 'Plane angles:'
        
        for plane, pcd in zip(planes, pcds):
            angle = plane.normal_angle_to_vector_xz(np.array([[0,0,1]])) * 180 / np.pi
            title += '{:.2f} '.format(angle)

            pcd.color_points_by_distance_plane(plane, threshold=0.15)
            plot = pcd.vis(ax, dim_2d=True, s=3, lim=[-20, 40, -12, 12])
            
            thresholds = [-np.inf, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, np.inf]
            pcd.color_points_by_distance_plane(plane, threshold=0.0, clip = 0.5, signed=True, lists=thresholds)
            colors = ['red','orange','yellow','green','blue','purple', 'gray'] #, 'pink', 'orange', 'cyan']
            plot = pcd.vis(ax2, dim_2d=True, lim=[-20, 40, -12, 12], cmap=clr.ListedColormap(colors))
            
            
            """ 
            # failed to visualize the side view, needs an orthogonal vector, otherwise very noisy
            ax2 = fig.add_subplot(212)
            thresholds = [-np.inf, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, np.inf]
            pcd.color_points_by_distance_plane(plane, threshold=0.0, clip = 0.5, signed=True, lists=thresholds)
            colors = ['red','orange','yellow','green','blue','purple', 'gray'] #, 'pink', 'orange', 'cyan']
            plot = pcd.vis(ax2, dim_2d=True, lim=[-20, 40, -5, 0], cmap=clr.ListedColormap(colors), side=True)
            ax2.set_title("threshold: -0.3, -0.15, -0.05, 0.05, 0.15, 0.3")
            plt.colorbar(plot)
            """
        plt.colorbar(plot)
        ax.set_title('[{:.2f},{:.2f}] boundary'.format(-0.15, 0.15))
        ax2.set_title("threshold: -0.3, -0.15, -0.05, 0.05, 0.15, 0.3")
        plt.colorbar(plot)
        plt.suptitle(title)
    else:
        ax = fig.add_subplot(111, projection='3d')
        plane.vis(ax)
        plot = pcd.vis(ax, dim_2d=False)
        plt.colorbar(plot)

def data_prepare(filenum = 1):
    lidar_pitch = 9.5/180*np.pi
    
    filename = "/home/henry/Documents/projects/avl/Detection/ground/data/points_data_"+repr(filenum)+".txt"
    planes, pcds = read_points_data(filename)
    
    tf = Rotation(roll=0.0, pitch=lidar_pitch, yaw=0.0)
    for plane in planes:
        plane.rotate_around_axis(axis="y", angle=-lidar_pitch)
    for pcd in pcds:
        pcd.rotate(tf)
    return planes, pcds

def test_estimation_single_plane(plane, pcd):
    # print(np.min(pcd.data[0,:]), np.max(pcd.data[0,:]))
    # vis(plane, pcd, dim_2d=True)

    # the onlyfloor points
    sac = RANSAC(Plane3D, min_sample_num=3, threshold=0.22, iteration=50, method="MSAC")
    vec1 = np.array([1,0,0])
    vec2 = np.array([0,0,1])
    pt1 = np.array([0,0,0])
    threshold = [6.0, -3.0]
    pcd_close, _ = clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold)
    plane2, _, _, _ = sac.ransac(pcd_close.data.T)
    vis(plane2, pcd, dim_2d=True)
    
    pcd = pcd_close

    normal = vec1.reshape([-1,1]) / np.linalg.norm(vec1)
    depth = (pcd.data.T @ normal).reshape([-1])
    distance = plane2.distance_to_plane(pcd.data.T)
    threshold_outer = 0.3
    threshold_inner = 0.1
    mask_outer = distance < threshold_outer
    mask_inner = distance < threshold_inner
    bin_dist = 5.0
    depth_min = np.min(depth)
    bin_num = int((np.max(depth) -  depth_min)/ bin_dist) + 1
    for i in range(bin_num):
        depth_thred_min, depth_thred_max = i*bin_dist+depth_min, (i+1)*bin_dist+depth_min
        mask_depth = np.logical_and(depth > depth_thred_min, depth < depth_thred_max)
        part_inner = np.logical_and(mask_depth, mask_inner)
        part_outer = np.logical_and(mask_depth, mask_outer)
        sum_outer = np.sum(part_outer)
        sum_inner = np.sum(part_inner)
        if sum_outer == 0:
            ratio = 1
        else:
            ratio = sum_inner / sum_outer
        if not ratio == 1:
            print(i, "{:.1f}".format(depth_thred_min), "{:.1f}".format(depth_thred_max), sum_inner, sum_outer, "{:.4f}".format(ratio))
    
    plt.show()

def test_estimation(plane, pcd):
    planes, pcds, distance_thresholds = [], [], []
    # print(np.min(pcd.data[0,:]), np.max(pcd.data[0,:]))
    # vis(plane, pcd, dim_2d=True)

    # the onlyfloor points
    sac = RANSAC(Plane3D, min_sample_num=3, threshold=0.22, iteration=5000, method="MSAC")
    vec1 = np.array([1,0,0])
    vec2 = np.array([0,0,1])
    pt1 = np.array([0,0,0])
    threshold = [6.0, -3.0]
    pcd_close, _ = clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold)
    plane2, _, _, _ = sac.ransac(pcd_close.data.T)
    vis(plane2, pcd, dim_2d=True)
    planes.append(plane2)

    normal = vec1.reshape([-1,1]) / np.linalg.norm(vec1)
    depth = (pcd_close.data.T @ normal).reshape([-1])
    distance = plane2.distance_to_plane(pcd_close.data.T)
    threshold_outer = 0.3
    threshold_inner = 0.1
    mask_outer = distance < threshold_outer
    mask_inner = distance < threshold_inner
    bin_dist = 5.0
    depth_min = np.min(depth)
    bin_num = int((np.max(depth) -  depth_min)/ bin_dist) + 1
    for i in range(bin_num):
        depth_thred_min, depth_thred_max = i*bin_dist+depth_min, (i+1)*bin_dist+depth_min
        mask_depth = np.logical_and(depth > depth_thred_min, depth < depth_thred_max)
        part_inner = np.logical_and(mask_depth, mask_inner)
        part_outer = np.logical_and(mask_depth, mask_outer)
        sum_outer = np.sum(part_outer)
        sum_inner = np.sum(part_inner)
        if sum_outer == 0:
            ratio = 1
        else:
            ratio = sum_inner / sum_outer
        if not ratio == 1:
            print(i, "{:.1f}".format(depth_thred_min), "{:.1f}".format(depth_thred_max), sum_inner, sum_outer, "{:.4f}".format(ratio))
        if ratio < 0.9 and sum_outer > 50:
            weight_dict = {'method':"x norm", 'param':{'x0':(depth_thred_max+depth_thred_min)/2.0, 'norm':2}}
            pcd_in, pcd_out = pcd_close.clip_by_x([depth_thred_min - 10, np.inf])
            plane_add, _, _, _ = sac.ransac(pcd_in.data.T, weight=weight_dict)
            vis(plane_add, pcd, dim_2d=True)
            planes.append(plane_add)
            pcds.append(pcd_in)
            distance_thresholds.append(depth_thred_min)
    
    plt.show()

def test_estimation_combine_planes(plane, pcd):
    planes, pcds, distance_thresholds = [], [], []
    # print(np.min(pcd.data[0,:]), np.max(pcd.data[0,:]))
    # vis(plane, pcd, dim_2d=True)

    # the reestimated plane
    sac = RANSAC(Plane3D, min_sample_num=3, threshold=0.22, iteration=50, method="MSAC")
    vec1 = np.array([1,0,0])
    vec2 = np.array([0,0,1])
    pt1 = np.array([0,0,0])
    threshold = [6.0, -3.0]
    pcd_close, _ = clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold)
    plane2, _, _, _ = sac.ransac(pcd_close.data.T)
    vis(plane2, pcd, dim_2d=True)
    planes.append(plane2)

    normal = vec1.reshape([-1,1]) / np.linalg.norm(vec1)
    depth = (pcd_close.data.T @ normal).reshape([-1])
    distance = plane2.distance_to_plane(pcd_close.data.T)
    threshold_outer = 0.3
    threshold_inner = 0.1
    mask_outer = distance < threshold_outer
    mask_inner = distance < threshold_inner
    bin_dist = 5.0
    depth_min = np.min(depth)
    bin_num = int((np.max(depth) -  depth_min)/ bin_dist) + 1
    for i in range(bin_num):
        depth_thred_min, depth_thred_max = i*bin_dist+depth_min, (i+1)*bin_dist+depth_min
        mask_depth = np.logical_and(depth > depth_thred_min, depth < depth_thred_max)
        part_inner = np.logical_and(mask_depth, mask_inner)
        part_outer = np.logical_and(mask_depth, mask_outer)
        sum_outer = np.sum(part_outer)
        sum_inner = np.sum(part_inner)
        if sum_outer == 0:
            ratio = 1
        else:
            ratio = sum_inner / sum_outer
        if not ratio == 1:
            print(i, "{:.1f}".format(depth_thred_min), "{:.1f}".format(depth_thred_max), sum_inner, sum_outer, "{:.4f}".format(ratio))
        if ratio < 0.9 and sum_outer > 50:
            weight_dict = {'method':"x norm", 'param':{'x0':(depth_thred_max+depth_thred_min)/2.0, 'norm':2}}
            pcd_close_in, pcd_close_out = pcd_close.clip_by_x([depth_thred_min - 10, np.inf])
            pcd_in, pcd_out = pcd.clip_by_x([depth_thred_min - 10, np.inf])
            plane_add, _, _, _ = sac.ransac(pcd_close_in.data.T, weight=weight_dict)
            planes.append(plane_add)
            if len(pcds) == 0:
                pcds.append(pcd_out)
            pcds.append(pcd_in)
            distance_thresholds.append(depth_thred_min)
    if len(pcds) == 0:
        pcds.append(pcd)
    vis_multiple(planes, pcds, dim_2d=True)

    plt.show()

def test_part_display():
    filename = "/home/henry/Documents/projects/avl/Detection/ground/data/points_raw_2.txt"
    pcd = read_points_raw(filename)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    for i in range(100):
        data = pcd.data[:,i*int(pcd.data.shape[1]/100):(i+1)*int(pcd.data.shape[1]/100)]
        ax.scatter(data[0,:], data[1,:], s=1)
        plt.pause(0.1)
    plt.show()

def test_scan_display():
    filename = "/home/henry/Documents/projects/avl/Detection/ground/data/points_raw_2.txt"
    pcd = read_points_raw(filename)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    for i in range(16):
        data = pcd.data[:,range(i, pcd.data.shape[1], 16)]
        ax.scatter(data[0,:], data[1,:])
        plt.pause(0.1)
    plt.show()

def test_all_pcd():
    for i in range(30, 61, 1):
        print("\nEstimating points data " + repr(i))
        planes, pcds = data_prepare(filenum=i)
        pcd = pcds[0]
        plane = planes[1]
        test_estimation(plane, pcd)

def test_cam_back_project():
    K = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02], 
                  [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    R = np.array([[0., -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]])
    C_world = np.array([[0, 0.5, 0]]).T
    t = -1 * R @ C_world
    cam = Camera(K, R, t)
    x = np.array([[1, 1000, 1000, 1000, 1000, 1000, 800, 600, 400, 604],
              [1, 500, 400, 300, 200, 300, 300, 300, 300, 300]])
    plane = Plane3D(0., 0., 1, 2)
    for i in range(x.shape[1]):
        d, C = cam.pixel_to_ray(x[0,i], x[1,i])
        intersection = plane.plane_ray_intersection(d, C)
        print(x[0,i],'\t', x[1,i],'\t', intersection[0,0],'\t', intersection[1,0],'\t', intersection[2,0])

def test_bounding_box_in_image():
    
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    
    bbox_data = np.loadtxt('data/bounding_boxes.txt')
    bboxes = [BoundingBox(box[0], box[1], box[2], box[3]) for box in bbox_data]
    for box in bboxes:
        box.vis(ax)
    K = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02], 
                  [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    R = np.array([[0., -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]])
    C_world = np.array([[0, 0.5, 0]]).T
    t = -1 * R @ C_world
    imSize = [1200, 400]
    cam = Camera(K, R, t, imSize=imSize)
    
    cam.show_image(ax)
    plt.title("image")
    plt.show()

def test_bounding_box_to_world():
    fig = plt.figure(figsize=(16,8))
    spec = gridspec.GridSpec(2,3)

    # large subplot
    # plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=1)
    ax = fig.add_subplot(spec[0,0:2])
    
    bbox_data = np.loadtxt('data/bounding_boxes.txt')
    bboxes = [BoundingBox(box[0], box[1], box[2], box[3]) for box in bbox_data]
    Ixlist, Iylist = [], []
    for box in bboxes:
        box.vis(ax)
        Ix, Iy = box.bottom_point()
        Ixlist.append(Ix)
        Iylist.append(Iy)
    ax.scatter(Ixlist, Iylist)

    K = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02], 
                  [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    R = np.array([[0., -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]])
    C_world = np.array([[0, 0.5, 0]]).T
    t = -1 * R @ C_world
    imSize = [1200, 400]
    cam = Camera(K, R, t, imSize=imSize)
    cam.show_image(ax)
    ax.axis('equal')
    plt.title("image")

    plane = Plane3D(0., 0., 1, 2)
    
    ax = fig.add_subplot(spec[1,:])
    xlist, ylist = [], []
    for bbox in bboxes:
        d, C = cam.bounding_box_to_ray(bbox)
        intersection = plane.plane_ray_intersection(d, C)
        print(intersection[0,0],'\t', intersection[1,0],'\t', intersection[2,0])
        xlist.append(intersection[0,0])
        ylist.append(intersection[1,0])
    ax.set_xlim([0, 80])
    ax.set_ylim([-40, 40])
    ax.scatter(xlist, ylist)
    plt.title("bird eye view")
    plt.suptitle("reproject bounding box to the world")

    plt.show()

# main
def main():
    
    planes, pcds = data_prepare(34)
    pcd = pcds[0]
    plane = planes[1]
    # test_scan_display()
    # test_part_display()
    # test_clip_pcd_by_distance_plane(pcd)
    # test_estimation(plane, pcd)
    # test_all_pcd()
    # test_cam_back_project()
    # test_bounding_box_in_image()
    # test_bounding_box_to_world()
    test_estimation_combine_planes(plane, pcd)
    
    


if __name__ == "__main__":
    main()
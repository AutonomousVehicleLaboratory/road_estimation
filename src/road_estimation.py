#!/usr/bin/env python
""" Estimate road surface from lidar scan, project bounding box on it.

Author: Henry Zhang
Date:February 11, 2020
"""

# module
from __future__ import absolute_import, division, print_function, unicode_literals
import rospy
import numpy as np
# import cv2
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion

from autoware_msgs.msg import DetectedObjectArray
from autoware_msgs.msg import DetectedObject


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec

from plane_3d import Plane3D
from point_cloud import PointCloud
from rotation import Quaternion as Quaternion_self
from ransac import RANSAC
from camera import Camera
from bounding_box import BoundingBox
from utils import clip_pcd_by_distance_plane
from vis import visualize_marker

np.set_printoptions(precision=3)

# parameters
global plane
global pub_intersect_markers
global pub_plane_markers
global pub_convex_hull_markers
# classes


# functions
def camera_setup_1():
    K = np.array([[1826.998004, 0., 1174.548672],
                  [0., 1802.603136, 776.028597],
                  [0., 0., 1. ]])
    """K = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02], 
                  [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    """
    R = np.array([[0., -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]])
    C_world = np.array([[0, 0.5, 0]]).T
    t = np.matmul(-1 * R , C_world)
    imSize = [1920, 1440]
    cam = Camera(K, R, t, imSize=imSize, id=1)
    return cam 

def camera_setup_6():
    K = np.array([[1790.634474, 0., 973.099292],
                  [0., 1785.950534, 803.294457],
                  [0., 0., 1. ]])
    """    
    K = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02], 
                  [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    """
    Rt = np.array([[ -2.1022535018250471e-01, -9.2112145235168197e-02, 9.7330398891652492e-01, -1.4076865278184414e-02],
                   [ -9.7735897207277012e-01, -4.6117027185500481e-03, -2.1153763709301088e-01, -3.1732881069183350e-01],
                   [ 2.3973774202277975e-02, -9.9573795995643932e-01, -8.9057134763516621e-02, -7.2184838354587555e-02],
                   [ 0., 0., 0., 1. ]])
    R = Rt[0:3, 0:3].T
    t = -np.matmul(R, Rt[0:3, 3:4])
    imSize = [1920, 1440]
    cam = Camera(K, R, t, imSize=imSize, id=6)
    return cam

def display_bboxes_in_world( camera, bboxes, ax1, ax2):
    global plane, pub_intersect_markers
    """
    ax1.cla()
    ax2.cla()
    Ixlist, Iylist = [], []
    for box in bboxes:
        box.vis(ax1)
        Ix, Iy = box.bottom_point()
        Ixlist.append(Ix)
        Iylist.append(Iy)
    ax1.scatter(Ixlist, Iylist)
    camera.show_image(ax1)
    
    # plane = Plane3D(0., 0., 1, 2)
    """
    vis_array = MarkerArray()
    xlist, ylist = [], []
    for box_id, bbox in enumerate(bboxes):
        d, C = camera.bounding_box_to_ray(bbox)
        intersection = plane.plane_ray_intersection(d, C)
        # print(intersection[0,0],'\t', intersection[1,0],'\t', intersection[2,0])
        marker = visualize_marker(intersection, mkr_id=box_id, scale=1, frame_id="velodyne")
        vis_array.markers.append(marker)
        xlist.append(intersection[0,0])
        ylist.append(intersection[1,0])
    """
    ax2.set_xlim([0, 80])
    ax2.set_ylim([-40, 40])
    ax2.scatter(xlist, ylist)
    plt.title("bird eye view")
    plt.suptitle("reproject bounding box to the world")
    """
    pub_intersect_markers.publish(vis_array)

    # plt.pause(0.001)

def bbox_array_callback(msg, args):
    camera, ax1, ax2 = args
    # rospy.loginfo("camera {:d} message received!!".format(camera.id))
    bboxes = []
    for obj in msg.objects:
        # rospy.loginfo("{}".format(obj.label))
        # rospy.loginfo("x:{} y:{} width:{} height:{} angle:{}".format(obj.x, obj.y, obj.width, obj.height, obj.angle))
        bbox = BoundingBox(obj.x, obj.y, obj.width, obj.height, obj.angle, label=obj.label)
        bboxes.append(bbox)
    if camera.id == 6:
        display_bboxes_in_world(camera, bboxes, ax1, ax2)

def estimate_plane(pcd):
    sac = RANSAC(Plane3D, min_sample_num=3, threshold=0.22, iteration=50, method="MSAC")
    vec1 = np.array([1,0,0])
    vec2 = np.array([0,0,1])
    pt1 = np.array([0,0,0])
    threshold = [6.0, -3.0]
    pcd_close, _ = clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold)
    plane1, _, _, _ = sac.ransac(pcd_close.data.T)
    # vis(plane1, pcd, dim_2d=True)

    normal = vec1.reshape([-1,1]) / np.linalg.norm(vec1)
    depth = np.matmul(pcd_close.data.T , normal).reshape([-1])
    distance = plane1.distance_to_plane(pcd_close.data.T)
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
            ratio = float(sum_inner) / sum_outer
        if not ratio == 1:
            print(i, "{:.1f}".format(depth_thred_min), "{:.1f}".format(depth_thred_max), sum_inner, sum_outer, "{:.4f}".format(ratio))
    
    print('Plane params:', plane1.param.T)
    return plane1

def create_and_publish_plane_markers(plane):
    v1 = np.array([[0, 0, 1.0]]).T
    v2 = np.array([[plane.a, plane.b, plane.c]]).T
    q_self = Quaternion_self.create_quaternion_from_vector_to_vector(v1, v2)
    q = Quaternion(q_self.x, q_self.y, q_self.z, q_self.w)
    marker_array = MarkerArray()
    marker = visualize_marker([10,0,(-plane.a * 10 - plane.d) / plane.c], 
                      frame_id="velodyne", 
                      mkr_type='cube', 
                      orientation=q, 
                      scale=[20,2,0.05])
    marker_array.markers.append(marker)
    return marker_array

def test_cam_back_project_convex_hull():
    global plane, pub_convex_hull_markers
    cam = camera_setup_6()

    x = np.array([[400, 300, 1100, 1000, 400],
                  [1150, 1200, 1200, 1150, 1150]])
    
    d_vec, C_vec = cam.pixel_to_ray_vec(x)
    intersection_vec = plane.plane_ray_intersection_vec(d_vec, C_vec)
    marker = visualize_marker([0,0,0], frame_id="velodyne", mkr_type="line_strip", scale=0.1, points=intersection_vec.T)
    pub_convex_hull_markers.publish(marker)

def pcd_callback(msg):
    global plane, pub_plane_markers
    print(msg.height, msg.width)
    pcd_original = np.empty((msg.width,3))
    for i, el in enumerate( pc2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True)):
        pcd_original[i,:] = el
    # pcd_original = [ i for i in ]
    # print(type(pcd_original[0][0]))
    # print(len(pcd_original))
    pcd = PointCloud(pcd_original.T)
    plane = estimate_plane(pcd)
    marker_array = create_and_publish_plane_markers(plane)
    pub_plane_markers.publish(marker_array)
    test_cam_back_project_convex_hull()

# main
def main():
    global pub_intersect_markers, pub_plane_markers, pub_convex_hull_markers
    rospy.init_node("road_estimation")
    
    fig = plt.figure(figsize=(16,8))
    spec = gridspec.GridSpec(2,3)
    ax1 = fig.add_subplot(spec[0,0:2])
    ax2 = fig.add_subplot(spec[1,:])
    
    cam1 = camera_setup_1()
    cam6 = camera_setup_6()

    # specify topic and data type
    sub_bbox_1 = rospy.Subscriber("camera1/detection/vision_objects", DetectedObjectArray, bbox_array_callback, [cam1, ax1, ax2])
    sub_bbox_6 = rospy.Subscriber("camera6/detection/vision_objects", DetectedObjectArray, bbox_array_callback, [cam6, ax1, ax2])
    sub_pcd = rospy.Subscriber("/points_raw", PointCloud2, pcd_callback)

    # Publisher
    pub_intersect_markers = rospy.Publisher("/vision_objects_position_rviz", MarkerArray, queue_size=10)
    pub_plane_markers = rospy.Publisher("/estimated_plane_rviz", MarkerArray, queue_size=10)
    pub_convex_hull_markers = rospy.Publisher("/estimated_convex_hull_rviz", Marker, queue_size=10)

    rospy.spin()
    # plt.show(block=True)

if __name__ == "__main__":
    main()
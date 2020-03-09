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
from shape_msgs.msg import Plane

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
from camera import camera_setup_1, camera_setup_6
from bounding_box import BoundingBox
from utils import clip_pcd_by_distance_plane
from utils_ros import create_point_cloud
from vis import visualize_marker

np.set_printoptions(precision=3)

# parameters
global plane
plane = Plane3D(-0.157, 0, 0.988, 1.9)
global pub_plane
global pub_intersect_markers
global pub_plane_markers
global pub_pcd_inlier
global pub_pcd_outlier
# classes


# functions

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
    global plane
    sac = RANSAC(Plane3D, min_sample_num=3, threshold=0.22, iteration=200, method="MSAC")
    vec1 = np.array([1,0,0])
    vec2 = np.array([0,0,1])
    pt1 = np.array([0,0,0])
    threshold = [6.0, -3.0]
    pcd_close, _ = clip_pcd_by_distance_plane(pcd, vec1, vec2, pt1, threshold)
    seed=0
    np.random.seed(seed)
    plane1, _, _, _ = sac.ransac(pcd_close.data.T, constraint=plane.param, constraint_threshold=0.5)
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
    threshold_inlier = 0.15
    pcd_inlier = pcd_close.data[:,distance <= threshold_inlier]
    pcd_outlier = pcd_close.data[:,distance > threshold_inlier]
    return plane1, pcd_inlier, pcd_outlier

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
                      scale=[20,2,0.05],
                       mkr_color = [0.0, 0.8, 0.0, 0.8])
    marker_array.markers.append(marker)
    return marker_array

def pcd_callback(msg):
    global plane, pub_plane_markers, pub_plane, pub_pcd_inlier, pub_pcd_outlier
    rospy.logwarn("pcd height:%d, width:%d", msg.height, msg.width)
    pcd_original = np.empty((msg.width,3))
    for i, el in enumerate( pc2.read_points(msg, field_names = ("x", "y", "z"), skip_nans=True)):
        pcd_original[i,:] = el
    # pcd_original = [ i for i in ]
    # print(type(pcd_original[0][0]))
    # print(len(pcd_original))
    pcd = PointCloud(pcd_original.T)
    plane, pcd_inlier, pcd_outlier = estimate_plane(pcd)
    marker_array = create_and_publish_plane_markers(plane)
    plane_msg = Plane()
    plane_msg.coef[0], plane_msg.coef[1], plane_msg.coef[2], plane_msg.coef[3] = plane.a, plane.b, plane.c, plane.d
    pub_plane_markers.publish(marker_array)
    pub_plane.publish(plane_msg)

    pcd_msg_inlier = create_point_cloud(pcd_inlier.T, frame_id='velodyne')
    pcd_msg_outlier = create_point_cloud(pcd_outlier.T, frame_id='velodyne')
    pub_pcd_inlier.publish(pcd_msg_inlier)
    pub_pcd_outlier.publish(pcd_msg_outlier)

    rospy.logwarn("Finished plane estimation")

# main
def main():
    global pub_intersect_markers, pub_plane_markers, pub_plane, pub_pcd_inlier, pub_pcd_outlier
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
    pub_plane = rospy.Publisher("/estimated_plane", Plane, queue_size = 10)
    pub_pcd_inlier = rospy.Publisher("/points_inlier", PointCloud2, queue_size=10)
    pub_pcd_outlier = rospy.Publisher("/points_outlier", PointCloud2, queue_size=10)

    rospy.spin()
    # plt.show(block=True)

if __name__ == "__main__":
    main()

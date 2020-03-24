#!/usr/bin/env python
""" Estimate road surface from lidar scan, project bounding box on it.

Author: Henry Zhang
Date:February 11, 2020
"""

# module
from __future__ import absolute_import, division, print_function, unicode_literals
import rospy
import time
import numpy as np
# import cv2
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, PoseStamped
from shape_msgs.msg import Plane
from tf import Transformer, TransformListener, TransformerROS
from tf.transformations import euler_from_quaternion

from autoware_msgs.msg import DetectedObjectArray
from autoware_msgs.msg import DetectedObject


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec

from plane_3d import Plane3D
from point_cloud import PointCloud, clip_pcd_by_distance_plane
from rotation import Quaternion as Quaternion_self
from ransac import RANSAC
from camera import camera_setup_1, camera_setup_6
from bounding_box import BoundingBox
from utils_ros import create_point_cloud, get_transformation, pointcloud2_to_xyz_array, xyz_array_to_pointcloud2
from vis import visualize_marker
from tracker import Tracker


import cProfile, pstats, io

np.set_printoptions(precision=3)

# parameters

# classes

# functions
def profile(fnc):
  """ A decorator that use cProfile to profile a function """
  def inner(*args, **argv):
    pr = cProfile.Profile()
    pr.enable()
    retval = fnc(*args, **argv)
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())
    return retval

  return inner

class RoadEstimation:
    def __init__(self):
        # specify topic and data type
        self.sub_bbox_1 = rospy.Subscriber("camera1/detection/vision_objects", DetectedObjectArray, self.bbox_array_callback)
        self.sub_bbox_6 = rospy.Subscriber("camera6/detection/vision_objects", DetectedObjectArray, self.bbox_array_callback)
        self.sub_pcd = rospy.Subscriber("/points_raw", PointCloud2, self.pcd_callback)
        # self.sub_pose = rospy.Subscriber("/current_pose", PoseStamped, self.pose_callback)

        # Publisher
        self.pub_intersect_markers = rospy.Publisher("/vision_objects_position_rviz", MarkerArray, queue_size=10)
        self.pub_plane_markers = rospy.Publisher("/estimated_plane_rviz", MarkerArray, queue_size=10)
        self.pub_plane = rospy.Publisher("/estimated_plane", Plane, queue_size = 10)
        self.pub_pcd_inlier = rospy.Publisher("/points_inlier", PointCloud2, queue_size=10)
        self.pub_pcd_outlier = rospy.Publisher("/points_outlier", PointCloud2, queue_size=10)
        
        self.cam1 = camera_setup_1()
        self.cam6 = camera_setup_6() 
        self.plane = Plane3D(-0.157, 0, 0.988, 1.9)
        self.plane_world = None
        self.plane_tracker = None
        self.sac = RANSAC(Plane3D, min_sample_num=3, threshold=0.22, iteration=200, method="MSAC")

        self.tf_listener = TransformListener()
        self.tfmr = Transformer()
        self.tf_ros = TransformerROS()

    def display_bboxes_in_world(self, cam, bboxes):
        vis_array = MarkerArray()
        xlist, ylist = [], []
        for box_id, bbox in enumerate(bboxes):
            d, C = cam.bounding_box_to_ray(bbox)
            intersection = self.plane.plane_ray_intersection(d, C)
            # print(intersection[0,0],'\t', intersection[1,0],'\t', intersection[2,0])
            marker = visualize_marker(intersection, mkr_id=box_id, scale=0.5, frame_id="velodyne", mkr_color = [0.0, 0.8, 0.0, 1.0])
            vis_array.markers.append(marker)
            xlist.append(intersection[0,0])
            ylist.append(intersection[1,0])

        self.pub_intersect_markers.publish(vis_array)

        # plt.pause(0.001)

    def bbox_array_callback(self, msg):
        if msg.header.frame_id == "camera1":
            cam = self.cam1
        elif msg.header.frame_id == "camera6":
            cam = self.cam6
        else:
            rospy.logwarn("unrecognized frame id {}, bounding box callback in road estimation fail", msg.header.frame_id)
            return

        # rospy.loginfo("camera {:d} message received!!".format(camera.id))
        bboxes = []
        for obj in msg.objects:
            # rospy.loginfo("{}".format(obj.label))
            # rospy.loginfo("x:{} y:{} width:{} height:{} angle:{}".format(obj.x, obj.y, obj.width, obj.height, obj.angle))
            bbox = BoundingBox(obj.x, obj.y, obj.width, obj.height, obj.angle, label=obj.label)
            bboxes.append(bbox)
        self.display_bboxes_in_world(cam, bboxes)

    def estimate_plane(self, pcd):
        # threshold_z = [2.0, -0.5]
        # pcd_z = clip_pcd_by_distance_plane(pcd, self.plane, threshold_z, in_only=True)

        vec1 = np.array([1,0,0])
        vec2 = np.array([0,0,1])
        pt1 = np.array([0,0,0])
        threshold = [-3.0, 6.0]
        plane_from_vec = Plane3D.create_plane_from_vectors_and_point(vec1, vec2, pt1)
        pcd_close = clip_pcd_by_distance_plane(pcd, plane_from_vec, threshold, in_only=True)

        # pcd_close = pcd_close.extract_low()
        
        seed=0
        np.random.seed(seed)
        plane1, _, _, _ = self.sac.ransac(pcd_close.data.T, constraint=self.plane.param, constraint_threshold=0.5)
        # vis(plane1, pcd, dim_2d=True)

        # normal = vec1.reshape([-1,1]) / np.linalg.norm(vec1)
        # depth = np.matmul(pcd_close.data.T , normal).reshape([-1])
        distance = plane1.distance_to_plane(pcd_close.data.T)
        # threshold_outer = 0.3
        # threshold_inner = 0.1
        # mask_outer = distance < threshold_outer
        # mask_inner = distance < threshold_inner
        # bin_dist = 5.0
        # depth_min = np.min(depth)
        # bin_num = int((np.max(depth) -  depth_min)/ bin_dist) + 1
        # for i in range(bin_num):
        #     depth_thred_min, depth_thred_max = i*bin_dist+depth_min, (i+1)*bin_dist+depth_min
        #     mask_depth = np.logical_and(depth > depth_thred_min, depth < depth_thred_max)
        #     part_inner = np.logical_and(mask_depth, mask_inner)
        #     part_outer = np.logical_and(mask_depth, mask_outer)
        #     sum_outer = np.sum(part_outer)
        #     sum_inner = np.sum(part_inner)
        #     if sum_outer == 0:
        #         ratio = 1
        #     else:
        #         ratio = float(sum_inner) / sum_outer
        #     if not ratio == 1:
        #         print(i, "{:.1f}".format(depth_thred_min), "{:.1f}".format(depth_thred_max), sum_inner, sum_outer, "{:.4f}".format(ratio))
        
        print('Plane params:', plane1.param.T)
        threshold_inlier = 0.15
        pcd_inlier = pcd_close.data[:,distance <= threshold_inlier]
        pcd_outlier = pcd_close.data[:,distance > threshold_inlier]
        return plane1, pcd_inlier, pcd_outlier

    def create_and_publish_plane_markers(self, plane, frame_id='velodyne', center=None, normal=None):
        if normal is None:
            v1 = np.array([[0, 0, 1.0]]).T
        else:
            v1 = normal
        v2 = np.array([[plane.a, plane.b, plane.c]]).T
        q_self = Quaternion_self.create_quaternion_from_vector_to_vector(v1, v2)
        q = Quaternion(q_self.x, q_self.y, q_self.z, q_self.w)
        euler = np.array(euler_from_quaternion((q.x, q.y, q.z, q.w))) * 180 / np.pi
        print("Euler: ", euler)
        marker_array = MarkerArray()
        if center is None:
            center = [10,0,(-plane.a * 10 - plane.d) / plane.c]
        marker = visualize_marker(center, 
                        frame_id=frame_id, 
                        mkr_type='cube', 
                        orientation=q, 
                        scale=[20,10,0.05],
                        lifetime=30,
                        mkr_color = [0.0, 0.8, 0.0, 0.8])
        marker_array.markers.append(marker)
        return marker_array
    
    # @profile # profiling for analysis
    def pcd_callback(self, msg):
        rospy.logwarn("Getting pcd at: %d.%09ds, (%d,%d)", msg.header.stamp.secs, msg.header.stamp.nsecs, msg.height, msg.width)

        pcd_original = pointcloud2_to_xyz_array(msg)

        pcd = PointCloud(pcd_original.T)
        
        self.plane, pcd_inlier, pcd_outlier = self.estimate_plane(pcd)
        transform_matrix, trans, rot, euler = get_transformation( frame_from='/velodyne', frame_to='/world',
                                                                  time_from= msg.header.stamp, time_to=msg.header.stamp, static_frame='/world',
                                                                  tf_listener=self.tf_listener, tf_ros=self.tf_ros)
        if not transform_matrix is None:
            plane_world_param = np.matmul( np.linalg.inv(transform_matrix).T, np.array([[ self.plane.a, self.plane.b, self.plane.c, self.plane.d]]).T)
            plane_world_param = plane_world_param / np.linalg.norm(plane_world_param[0:3])
            
            if self.plane_tracker is None:
                self.plane_tracker = Tracker(msg.header.stamp, plane_world_param)
            else:
                self.plane_tracker.predict(msg.header.stamp)
                self.plane_tracker.update(plane_world_param)
            print("plane_world:", plane_world_param.T)
            print("plane_traker:", self.plane_tracker.filter.x_post.T)
            
            # self.plane_world = Plane3D(plane_world_param[0,0], plane_world_param[1,0], plane_world_param[2,0], plane_world_param[3,0])
            self.plane_world = Plane3D(self.plane_tracker.filter.x_post[0,0], 
                                       self.plane_tracker.filter.x_post[1,0], 
                                       self.plane_tracker.filter.x_post[2,0], 
                                       self.plane_tracker.filter.x_post[3,0])
            center_pos = np.matmul(transform_matrix, np.array([[10, 0, (-self.plane.a * 10 - self.plane.d) / self.plane.c, 1]]).T)
            center_pos = center_pos[0:3].flatten()
            # normal = np.matmul( transform_matrix, np.array([[0., 0., 1., 0.]]).T)
            # normal = normal[0:3]
            normal = None
            marker_array = self.create_and_publish_plane_markers(self.plane_world, frame_id='world', center=center_pos, normal=normal)
            self.pub_plane_markers.publish(marker_array)
        
        plane_msg = Plane()
        plane_msg.coef[0], plane_msg.coef[1], plane_msg.coef[2], plane_msg.coef[3] = self.plane.a, self.plane.b, self.plane.c, self.plane.d
        
        self.pub_plane.publish(plane_msg)

        # pcd_msg_inlier = create_point_cloud(pcd_inlier.T, frame_id='velodyne')
        # pcd_msg_outlier = create_point_cloud(pcd_outlier.T, frame_id='velodyne')
        pcd_msg_inlier = xyz_array_to_pointcloud2(pcd_inlier.T, stamp=msg.header.stamp, frame_id='velodyne')
        pcd_msg_outlier = xyz_array_to_pointcloud2(pcd_outlier.T, stamp=msg.header.stamp, frame_id='velodyne')
        self.pub_pcd_inlier.publish(pcd_msg_inlier)
        self.pub_pcd_outlier.publish(pcd_msg_outlier)

        rospy.logwarn("Finished plane estimation")

    def pose_callback(self, msg):
        rospy.logdebug("Getting pose at: %d.%09ds", msg.header.stamp.secs, msg.header.stamp.nsecs)
        # print("Pose: position:", msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        # print("Pose: orientation:", msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w )    
        # transform_matrix, trans, rot, euler = get_transformation( frame_from='/base_link', frame_to='/velodyne',
        #                                                           time_from= rospy.Time(0), time_to=rospy.Time(0), static_frame='/world',
        #                                                           tf_listener=self.tf_listener, tf_ros=self.tf_ros)
        # print("baselink to localizer:", euler)

# main

def main():
    rospy.init_node("road_estimation")

    re = RoadEstimation()

    rospy.spin()
    # plt.show(block=True)

if __name__ == "__main__":
    main()

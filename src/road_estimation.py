#!/usr/bin/env python
""" Estimate road surface from lidar scan, project bounding box on it.

Author: Henry Zhang
Date:February 11, 2020
"""

# module
import rospy
import numpy as np
# import cv2
# from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import String
from autoware_msgs.msg import DetectedObjectArray
from autoware_msgs.msg import DetectedObject
# from visualization_msgs.msg import MarkerArray
# from visualization_msgs.msg import Marker

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.colors as clr
import matplotlib.gridspec as gridspec

from plane_3d import Plane3D
# from rotation import Rotation
# from ransac import RANSAC
from camera import Camera
from bounding_box import BoundingBox

np.set_printoptions(precision=3)

# parameters


# classes


# functions
def camera_setup_1():
    K = np.array([[1.6616804529944393e+03, 0., 9.3787145705318176e+02],
                  [0., 1.7023305136400143e+03, 7.3850001899950701e+02],
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
    K = np.array([[1.6616804529944393e+03, 0., 9.3787145705318176e+02],
                  [0., 1.7023305136400143e+03, 7.3850001899950701e+02],
                  [0., 0., 1. ]])
    """    
    K = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02], 
                  [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    """
    R = np.array([[0., -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]])
    C_world = np.array([[0, 0.5, 0]]).T
    t = np.matmul(-1 * R , C_world)
    imSize = [1920, 1440]
    cam = Camera(K, R, t, imSize=imSize, id=6)
    return cam 
    """cam.show_image(ax)
    ax.axis('equal')
    plt.title("image")
    """
def display_bboxes_in_world( camera, bboxes, ax1, ax2):
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

    plane = Plane3D(0., 0., 1, 2)
    
    
    xlist, ylist = [], []
    for bbox in bboxes:
        d, C = camera.bounding_box_to_ray(bbox)
        intersection = plane.plane_ray_intersection(d, C)
        print(intersection[0,0],'\t', intersection[1,0],'\t', intersection[2,0])
        xlist.append(intersection[0,0])
        ylist.append(intersection[1,0])
    ax2.set_xlim([0, 80])
    ax2.set_ylim([-40, 40])
    ax2.scatter(xlist, ylist)
    plt.title("bird eye view")
    plt.suptitle("reproject bounding box to the world")

    plt.pause(0.001)

def bbox_array_callback(msg, args):
    camera, ax1, ax2 = args
    rospy.loginfo("camera {:d} message received!!".format(camera.id))
    bboxes = []
    for obj in msg.objects:
        rospy.loginfo("{}".format(obj.label))
        rospy.loginfo("x:{} y:{} width:{} height:{} angle:{}".format(obj.x, obj.y, obj.width, obj.height, obj.angle))
        bbox = BoundingBox(obj.x, obj.y, obj.width, obj.height, obj.angle, label=obj.label)
        bboxes.append(bbox)
    if camera.id == 1:
        display_bboxes_in_world(camera, bboxes, ax1, ax2)
    

# main
def main():
    
    
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

    #rospy.spin()
    plt.show(block=True)

if __name__ == "__main__":
    main()
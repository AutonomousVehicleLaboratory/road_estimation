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

# parameters


# classes


# functions

def bbox_array_callback(msg, camera_id):
    rospy.loginfo("camera {:d} message received!!".format(camera_id))
    for obj in msg.objects:
        rospy.loginfo("{}".format(obj.label))
        rospy.loginfo("x:{} y:{} width:{} height:{} angle:{}".format(obj.x, obj.y, obj.width, obj.height, obj.angle))

# main
def main():
    rospy.init_node("road_estimation")

    # specify topic and data type
    sub_bbox_1 = rospy.Subscriber("camera1/detection/vision_objects", DetectedObjectArray, bbox_array_callback, 1)
    sub_bbox_6 = rospy.Subscriber("camera6/detection/vision_objects", DetectedObjectArray, bbox_array_callback, 6)

    rospy.spin()

if __name__ == "__main__":
    main()
""" Visualization for

Author: Henry Zhang
Date:February 13, 2020
"""

# module
import rospy
from visualization_msgs.msg import Marker

# parameters


# classes


# functions
def visualize_marker(point, marker_arr, x_id=0, frame_id="base_link", mkr_type="sphere"):
    marker = Marker()
    marker.header.frame_id = frame_id.encode("ascii", "ignore")
    marker.header.stamp = rospy.get_rostime()

    # Color of Marker
    marker.color.a = 0.8
    marker.color.r = 0.2
    marker.color.g = 1
    marker.color.b = 0.2

    # Type of Marker
    
    if mkr_type == "triangle":
        marker.type = marker.TRIANGLE_LIST
    elif mkr_type == "arrow":
        marker.type = marker.ARROW
    else:
        marker.type = marker.SPHERE
    # Location of Marker
    marker.pose.position.x = point[0] 
    marker.pose.position.y = point[1]
    marker.pose.position.z = point[2]

    # Scale of Marker
    marker.scale.x = 3
    marker.scale.y = 3
    marker.scale.z = 3

    # Marker ID
    marker.id = x_id 

    marker_arr.markers.append(marker)

# main
def main():
    pass

if __name__ == "__main__":
    main()
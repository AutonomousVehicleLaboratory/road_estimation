""" Visualization for

Author: Henry Zhang
Date:February 13, 2020
"""

# module
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# parameters


# classes


# functions
def visualize_marker(point, mkr_id=0, frame_id="base_link", mkr_type="sphere", orientation = None, scale = 0.1, points = None):
    """ create a marker 
    
    Params:
        point: list of 3 coordinates
        mkr_id: unique int of marker id, default to 0, same will overwrite
        frame_id: str, frame, default to 'base_link'
        mkr_type: str, default to 'sphere', can be 'cube', 'line_strip', 'arrow', 'triangle'
        orientation: geometry_msgs.msg Quaternion
        scale: int, float, or 3 list, specify the scale of each dimension, default to 0.1
        points: n by 3 list if provided, default to None
    Return:
        constructed marker
    """
    marker = Marker()
    marker.header.frame_id = frame_id.encode("ascii", "ignore")
    marker.header.stamp = rospy.get_rostime()
    
    # Marker ID
    marker.id = mkr_id # each marker in the array needs to be assign to a differen id
    marker.lifetime.secs = 1

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
    elif mkr_type == "cube":
        marker.type = marker.CUBE
    elif mkr_type == "line_strip":
        marker.type = marker.LINE_STRIP
    else:
        marker.type = marker.SPHERE

    # Location of Marker
    marker.pose.position.x = point[0] 
    marker.pose.position.y = point[1]
    marker.pose.position.z = point[2]
    if not orientation is None:
        marker.pose.orientation = orientation

    # Scale of Marker
    if type(scale) == int or type(scale) == float:
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
    elif len(scale) == 3:
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
    else:
        print("unexpected scale for marker!")
        exit()

    if mkr_type == "line_strip":
        for pti in points:
            pt = Point()
            pt.x = pti[0]
            pt.y = pti[1]
            pt.z = pti[2]
            marker.points.append(pt)
    
    return marker

# main
def main():
    pass

if __name__ == "__main__":
    main()
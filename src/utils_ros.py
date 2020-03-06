""" Ros utils

Author: Henry Zhang
Date:February 26, 2020
"""

# module
import rospy
import numpy as np
import struct

from tf import TransformBroadcaster, TransformListener, TransformerROS, LookupException, ConnectivityException, ExtrapolationException
from tf.transformations import quaternion_matrix, euler_from_quaternion, euler_matrix
import tf_conversions
from geometry_msgs.msg import TransformStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# parameters


# classes


# functions

pub = rospy.Publisher("point_cloud2", PointCloud2, queue_size=2)


def create_point_cloud(xyz, rgb=None, frame_id='world'):
    """ prepare point cloud 

    Params:
        xyz: n by 3 float numbers
        rgb: n by 4 array
    return:
        point_cloud_msg
    """
    header = Header()
    header.frame_id = frame_id
    if rgb is None:
        points = xyz.tolist()
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)
                  ]
    else:
        points = xyz.tolist()
        for i in range(xyz.shape[0]):
            rgba_i = struct.unpack('I', struct.Struct('BBBB').pack(int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2]), 255))[0]
            points[i].append(rgba_i)
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgba', 12, PointField.UINT32, 1)
                  ]
    pc2 = point_cloud2.create_cloud(header, fields, points)
    return pc2


def set_map_pose(pose, parent_frame_id, child_frame_id):
    br = TransformBroadcaster()
    m = TransformStamped()
    m.header.frame_id = parent_frame_id
    m.header.stamp = rospy.Time.now()
    m.child_frame_id = child_frame_id
    m.transform.translation.x = pose.position.x
    m.transform.translation.y = pose.position.y
    m.transform.translation.z = pose.position.z
    m.transform.rotation.x = pose.orientation.x
    m.transform.rotation.y = pose.orientation.y
    m.transform.rotation.z = pose.orientation.z
    m.transform.rotation.w = pose.orientation.w
    br.sendTransformMessage(m)

def get_transformation(frame_from='/base_link', frame_to='/local_map',
                       time_from=None, time_to=None, static_frame=None,
                       tf_listener = None, tf_ros = None):
    if tf_listener is None:
        tf_listener = TransformListener()
    if tf_ros is None:
        tf_ros = TransformerROS()
    try:
        if time_from is None or time_to is None:
            # tf_listener.waitForTransform(frame_from, frame_to, rospy.Time(), rospy.Duration(1.0))
            (trans, rot) = tf_listener.lookupTransform(frame_to, frame_from, rospy.Time(0))
        else:
            # tf_listener.waitForTransformFull(frame_from, time_from, frame_to, time_to, static_frame, rospy.Duration(1.0))
            (trans, rot) = tf_listener.lookupTransformFull(frame_to, time_to, frame_from, time_from, static_frame)
    except (LookupException, ConnectivityException, ExtrapolationException):
        rospy.logerr("exception, from %s to %s frame may not have setup!", frame_from, frame_to)
        return None, None, None, None
    # pose.pose.orientation.w = 1.0    # Neutral orientation
    # tf_pose = self.tf_listener_.transformPose("/world", pose)
    # R_local = quaternion_matrix(tf_pose.pose.orientation)
    T = tf_ros.fromTranslationRotation(trans, rot)
    euler = euler_from_quaternion(rot)
    
    # print "Position of the pose in the local map:"
    # print trans, euler
    return T, trans, rot, euler

def get_transform_from_pose( pose, tf_ros=None):
    """ from pose to origin (assumed 0,0,0) """
    if tf_ros is None:
        tf_ros = TransformerROS()
    translation = ( pose.position.x, pose.position.y, pose.position.z)
    rotation = ( pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    T_pose_to_origin = tf_ros.fromTranslationRotation(translation, rotation)
    return T_pose_to_origin

def get_normal_from_pose(pose):
    """ from pose, compute the normal on z
    https://answers.ros.org/question/222101/get-vector-representation-of-x-y-z-axis-from-geometry_msgs-pose/?answer=222179#post-id-222179
    """
    # p = Pose()
    # p.orientation = pose.orientation
    # z1 = (quaternion_matrix((p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)))[0:3,2:3]
    z = tf_conversions.fromMsg(pose).M.UnitZ()
    normal = np.array([[z[0], z[1], z[2]]]).T
    
    return normal

# main
def main():
    pass

if __name__ == "__main__":
    main()
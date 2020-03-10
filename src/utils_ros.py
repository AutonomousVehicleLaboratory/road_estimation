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

def xyz_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12*points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg

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


DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]

pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}


def fields_to_dtype(fields, point_step):
    '''
    Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0  
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list

def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' Converts a rospy PointCloud2 message to a numpy recordarray
    Reshapes the returned array to have shape (height, width), even if the height is 1.
    The reason for using np.fromstring rather than struct.unpack is speed... especially
    for large point clouds, this will be <much> faster.
    '''
    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def get_xyz_points(cloud_array, remove_nans=True):
    '''
    Pulls out x, y, and z columns from the cloud recordarray, and returns a 3xN matrix.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values
    points = np.zeros(list(cloud_array.shape) + [3], dtype=np.float)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']

    return points

def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
    """
    super fast point cloud message extraction

    https://github.com/dlaz/pr2-python-minimal/blob/3fc1dad91d2029ddc62c0b8303c895545ea687f4/src/pr2_python/pointclouds.py
    """
    return get_xyz_points(pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)

# main
def main():
    pass

if __name__ == "__main__":
    main()
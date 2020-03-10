""" rotation

Author: Henry Zhang
Date:January 27, 2020
"""

# module
import numpy as np

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Quaternion as Quaternion_ros
from utils import cross
# parameters


# classes


class Quaternion:
    def __init__(self, x=0, y=0, z=0, w=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
        self.normalize()

    @classmethod
    def create_quaternion_from_vector_to_vector(cls, v1, v2):
        """ from two direction vectors, create the quaternion 
        formulas from https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
        verified by create axis_angle then convert to quaternion
        Params:
            v1, v2: direction vectors, expected 3 by 1
        Return:
            q: Quaternion class
        """
        q = cls()
        direction = cross(v1.reshape([-1]), v2.reshape([-1]))
        q.x = direction[0]
        q.y = direction[1]
        q.z = direction[2]
        q.w = np.sqrt((np.linalg.norm(v1) ** 2) * (np.linalg.norm(v2) ** 2)) + np.matmul(v1.T, v2).item()
        q.normalize()
        return q
    
    # @classmethod
    # def to_ros_quaternion(cls, x, y, z, w):
    #     q = Quaternion_ros()
    #     q.x = x
    #     q.y = y
    #     q.z = z
    #     q.w = w
    #     return q
    
    def to_ros_quaternion(self):
        q = Quaternion_ros()
        q.x = self.x
        q.y = self.y
        q.z = self.z
        q.w = self.w
        return q


    def normalize(self):
        norm = np.sqrt(np.sum(np.array([self.x, self.y, self.z, self.w])**2))
        self.x /= norm
        self.y /= norm
        self.z /= norm
        self.w /= norm
    
    def to_list_xyzw(self):
        orientation = [self.x, self.y, self.z, self.w]
        return orientation

class AxisAngle:
    def __init__(self, w):
        self.w = w
        self.angle = np.linalg.norm(w)
        self.axis = np.array(w) / self.angle
    
    @classmethod
    def create_axisangle_from_vector_to_vector(cls, v1, v2):
        direction = np.cross(v1.reshape([-1]), v2.reshape([-1]))
        axis = direction / np.linalg.norm(direction)
        angle = np.arccos( np.matmul(v1.reshape((3,1)).T, v2.reshape((3,1))).item() / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        w = axis * angle
        return cls(w)
    
    def to_quaternion(self):
        w = np.cos(self.angle / 2.0)
        direction = np.sin(self.angle / 2.0) * self.axis
        x = direction[0]
        y = direction[1]
        z = direction[2]
        return Quaternion(x, y, z, w)
        

class Rotation:
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
    
    def getMatrix(self):
        m_roll = np.array([[1, 0, 0],
                           [0, np.cos(self.roll), -np.sin(self.roll)],
                           [0, np.sin(self.roll), np.cos(self.roll)]])
        m_pitch = np.array([[np.cos(self.pitch), 0, np.sin(self.pitch)],
                           [0, 1, 0],
                           [-np.sin(self.pitch), 0, np.cos(self.pitch)]])
        m_yaw = np.array([[np.cos(self.yaw), -np.sin(self.yaw), 0],
                           [np.sin(self.yaw), np.cos(self.yaw), 0],
                           [0, 0, 1]])
        m_euler = np.matmul( np.matmul(m_yaw, m_pitch),  m_roll)
        return m_euler
    
    

# functions

def create_euler_from_vectors(v1, v2):
    # q = Quaternion.create_quaternion_from_vector_to_vector(v1, v2)
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    direction = cross(v1, v2)
    x = direction[0]
    y = direction[1]
    z = direction[2]
    w = np.sqrt((v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]) * (v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2])) + \
        (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])
    # explicit_q = [q.x, q.y, q.z, q.w]
    euler = euler_from_quaternion([x,y,z,w])
    euler = np.array(euler).reshape(3,1)
    return euler


def test_vector_to_vector():
    v1 = np.array([[0, 0, 1]]).T
    v2 = np.array([[-0.5, 0, 1.732/2.0]]).T
    q = Quaternion.create_quaternion_from_vector_to_vector(v1, v2)
    rot = AxisAngle.create_axisangle_from_vector_to_vector(v1, v2)
    q2 = rot.to_quaternion()
    explicit_q = [q.x, q.y, q.z, q.w]
    euler = euler_from_quaternion(explicit_q)
    euler = np.array(euler).reshape(3,1)
    pass

def test_get_matrix():
    rot = Rotation(roll=9.0/180*np.pi)
    mat = rot.getMatrix()
    print(mat)

# main
def main():
    test_vector_to_vector()
    test_get_matrix()

if __name__ == "__main__":
    main()
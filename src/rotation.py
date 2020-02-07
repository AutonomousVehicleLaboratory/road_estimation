""" rotation operation

Author: Henry Zhang
Date:January 27, 2020
"""

# module
import numpy as np

# parameters


# classes
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
        m_euler = m_yaw @ m_pitch @ m_roll
        return m_euler

# functions
def test_get_matrix():
    rot = Rotation(roll=9.0/180*np.pi)
    mat = rot.getMatrix()
    print(mat)

# main
def main():
    test_get_matrix()

if __name__ == "__main__":
    main()
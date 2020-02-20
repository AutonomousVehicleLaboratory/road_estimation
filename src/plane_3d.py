""" Plane

Author: Henry Zhang
Date:January 25, 2020
"""

# module
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# parameters


# classes
class Plane3D:
    def __init__(self, a=0., b=0., c=0., d=0., weight={'method':"x norm", 'param':{'x0':0.0, 'norm':1}}):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.weight = weight
        self.normalize()
    
    @classmethod
    def create_plane_from_list(cls, param):
        return cls(param[0], param[1], param[2], param[3])
    
    @classmethod
    def create_plane_from_vectors_and_point(cls, vec1, vec2, pt1):
        """ create plane from two vectors and a point. 
        
        Params: 
            vec1, vec2, pt1: 3*1 array
        Return:
            plane: Plane3D """
        pt2 = vec1 / np.linalg.norm(vec1) + pt1
        pt3 = vec2 / np.linalg.norm(vec2) + pt1
        pts = np.vstack([pt1, pt2, pt3])
        return cls.fit(pts, method="min")
    
    @classmethod
    def fit(cls, data, method="least_square", weight=None):
        if method == "min":
            if len(data) == 3 and not np.all(data[0,:]-data[1,:]==0):
                a = (data[0,1] - data[1,1]) * (data[2,2] - data[1,2]) - (data[2,1] - data[1,1]) * (data[0,2] - data[1,2])
                b = (data[0,2] - data[1,2]) * (data[2,0] - data[1,0]) - (data[2,2] - data[1,2]) * (data[0,0] - data[1,0])
                c = (data[0,0] - data[1,0]) * (data[2,1] - data[1,1]) - (data[2,0] - data[1,0]) * (data[0,1] - data[1,1])
                d = -a*data[1,0] -b*data[1,1] - c*data[1,2]
            else:
                print("Error: Incorrect data, two required for minimal model fitting.")
                exit()
        elif method == "least_square":
            raise NotImplementedError
        else:
            raise NotImplementedError
        if weight is None:
            obj = cls(a, b, c, d)
        else:
            obj = cls(a, b, c, d, weight=weight)
        return obj
    
    def eval(self, data):
        if self.weight['method'] == "x norm":
            if self.weight['param']['norm'] == 1:
                x_norm = np.abs(data[:,0] - self.weight['param']['x0'])
            elif self.weight['param']['norm'] == 2:
                x_norm = (data[:,0] - self.weight['param']['x0'])**2
            else:
                raise NotImplementedError
            x_distance_recip = 1 / (x_norm + 1)
            x_distance_weight = x_distance_recip / np.max(x_distance_recip)
            cost = self.distance_to_plane(data) * x_distance_weight
        elif self.weight['method'] == "none":
            cost = self.distance_to_plane(data)
        else:
            raise NotImplementedError
        return cost

    def distance_to_plane(self, data):
        norm = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        if norm > 1e-3:
            distance = np.abs( np.matmul(data , self.param[0:3,:] ) + self.d).reshape([-1]) / norm
        else:
            distance = np.ones((data.shape[0])) * np.inf
        return distance
    
    def distance_to_plane_signed(self, data):
        norm = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        if norm > 1e-3:
            distance = ( np.matmul(data , self.param[0:3,:] ) + self.d).reshape([-1]) / norm
        else:
            distance = ( np.matmul(data , self.param[0:3,:] ) + self.d).reshape([-1]) * np.inf
        return distance

    def normalize(self):
        s = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        if s == 0:
            print("Error: a = 0, b = 0 and c = 0!!!, skip this result")
            return
        else:
            if self.c < 0:
                s = -1 * s
            self.a, self.b, self.c, self.d = self.a/s, self.b/s, self.c/s, self.d/s
            self.param = np.array([[self.a, self.b, self.c, self.d]]).T
    
    def rotate_around_axis(self, axis, angle):
        if axis == "y":
            norm = np.sqrt(self.a**2 + self.c**2) 
            theta = np.arctan2(self.c, self.a)
            theta_2 = theta + angle
            self.a, self.c = np.cos(theta_2)*norm, np.sin(theta_2)*norm
        
        self.param = np.array([[self.a, self.b, self.c, self.d]]).T
    
    def normal_angle_to_vector(self, vector):
        """ return the angle between the normal vector of the plane and another given vector

        Param:
            vector: 3 by 1 vector
        return:
            angle: a scaler represents the angle
        """
        vector = vector.reshape([3,1]) / np.linalg.norm(vector)
        self.normalize()
        angle = np.arccos( np.matmul(vector.T , self.param[0:3,:]) )
        return angle[0,0]
    
    def normal_angle_to_vector_xz(self, vector):
        """ return the angle between the normal vector of the plane and another given vector along xz plane

        Param:
            vector: 3 by 1 vector
        return:
            angle: a scaler represents the angle
        """
        vector = vector.reshape([3,1])
        innner = (vector[0,0]*self.a + vector[2,0]*self.c)
        scaling = np.sqrt(vector[0,0]**2 + vector[2,0]**2) * np.sqrt(self.a**2 + self.c**2)
        angle = np.arccos(innner / scaling)
        return angle
    
    def plane_ray_intersection(self, d, C):
        lam = (-1* np.matmul(self.param[0:3,:].T , C ) - self.d) / ( np.matmul(self.param[0:3,:].T , d) )
        intersection = d*lam + C
        return intersection

    def plane_ray_intersection_vec(self, d, C):
        param = np.array([[self.a, self.b, self.c]])
        k = (-self.d - np.matmul(param, C).item()) / np.matmul(param, d)
        intersection = k * d + C
        return intersection

    def vis(self, ax):
        if self.c != 0:
            xx, yy = np.meshgrid(range(11), range(11))
            z = (-self.a * xx - self.b * yy - self.d) * 1. / self.c
            ax.plot_surface(xx, yy, z, alpha=0.2)

# functions
def test_plane_3d():
    plane = Plane3D(1,1,1,-10)
    plane_2 = Plane3D(1,1,1,-10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plane.vis(ax)
    plane_2.rotate_around_axis(axis="y", angle=9.0/180*np.pi)
    ax.scatter([0, 0, 0, 10, 5], [0, 0, 10, 0, 0], [0, 10, 0, 0, 0])
    plane_2.vis(ax)
    plt.show()

def test_plane_fit():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    data = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
    plane2 = Plane3D.fit(data, method="min")
    ax.scatter(data[:,0], data[:,1], data[:,2])
    plane2.vis(ax)

    plt.show()

def test_create_plane_from_vectors_and_point():
    vec1 = np.array([0,10,5])
    vec2 = np.array([1,0,1])
    pt1 = np.array([0,0,0])
    plane = Plane3D.create_plane_from_vectors_and_point(vec1, vec2, pt1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plane.vis(ax)
    plt.show()

def test_normal_angle():
    plane1 = Plane3D(0,0,1,0)
    plane2 = Plane3D(0.2, 0, 1, 0)
    plane3 = Plane3D(1, 0, 1.732, 0)
    plane4 = Plane3D(0.1, 0.3, 1, 0)
    plane_list = [plane1, plane2, plane3, plane4]
    
    vector = np.array([[0,0,1.0]]).T
    
    for plane in plane_list:
        print(plane.normal_angle_to_vector(vector) * 180 / np.pi, end=" ")
    
    print(" ")

    for plane in plane_list:
        print(plane.normal_angle_to_vector_xz(vector) * 180 / np.pi, end=" ")


# main
def main():
    test_plane_3d()
    test_plane_fit()
    test_create_plane_from_vectors_and_point()
    test_normal_angle()

if __name__ == "__main__":
    main()
""" Projective Camera

Author: Henry Zhang
Date:February 03, 2020
"""

# module
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from plane_3d import Plane3D
from bounding_box import BoundingBox
from utils import homogenize, dehomogenize

# parameters


# classes
class Camera:
    def __init__(self, K, R, t, imSize = None, id=0):
        self.id = id
        self.K = K
        self.R = R
        self.t = t
        self.P = np.matmul(K, np.concatenate([R, t], axis=1)) # camera projection matrix (world to image)
        self.K_inv = np.linalg.inv(self.K)      # inverse of intrisic for convenience
        self.C_world_inhomo =np.matmul( - R.T , t)  # camera center in the world coordinate using inhomogeneous representation
        self.imSize = imSize                    # image size

    def pixel_to_ray(self, Ix, Iy, world=True):
        """ given a pixel, calculate a line that all points will be projected to the pixel
        through the camera
        Params:
            Ix - x coordinates in the image frame
            Iy - y coordinates in the image frame
        Return:
            line - 3D line """
        x = np.array([[Ix, Iy, 1.0]]).T
        if world == True:
            X_world_inhomo = np.matmul(self.R.T ,(np.matmul(self.K_inv , x)  - self.t))
            d = X_world_inhomo - self.C_world_inhomo
            d = d / np.sign(d[0,0]) / np.linalg.norm(d)
            C = self.C_world_inhomo
        else:
            X_cam_inhomo =np.matmul( self.K_inv , x) 
            d = X_cam_inhomo / np.sign(X_cam_inhomo[2,0]) / np.linalg.norm(X_cam_inhomo)
            C = np.zeros([3,1])
        
        return d, C
    
    def bounding_box_to_ray(self, bbox):
        Ix, Iy = bbox.bottom_point()
        d, C = self.pixel_to_ray(Ix, Iy, world=True)
        return d, C
    
    def show_image(self, ax):
        if self.imSize is None:
            print("imSize not set for this camera, cannot show image")
        else:
            ax.set_xlim([0, self.imSize[0]])
            ax.set_ylim([0, self.imSize[1]])
            ax.invert_yaxis()
    
    def get_image_coordinate(self, X):
        """ get the image coordinate of world point X """
        x_homo = np.matmul(self.P, homogenize(X))
        x = dehomogenize(x_homo)
        return x

            
        
# functions
def test_pixel_to_ray(cam):
    x = np.array([[1, 500, 1000],
              [1, 200, 400]])
    for i in range(x.shape[1]):
        d, C = cam.pixel_to_ray(x[0,i], x[1,i])

# main
def main():
    K = np.array([[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02], 
                  [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02],
                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
    R = np.array([[0., -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]])
    C_world = np.array([[0, 0.5, 0]]).T
    t = np.matmul( -1 * R , C_world) 
    imSize = [1200, 400]
    cam = Camera(K, R, t, imSize=imSize)
    test_pixel_to_ray(cam)
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    cam.show_image(ax)
    plt.title("image")
    plt.show()


if __name__ == "__main__":
    main()
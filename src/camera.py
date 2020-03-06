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
from utils import homogenize, dehomogenize, parameterize_rotation

# parameters


# classes
class Camera:
    def __init__(self, K, R, t, imSize = None, id=0, dist=None):
        """ setup camera

        Params
            K - 3 by 3 array, camera intrinsic matrix
            R - 3 by 3 array, camera extrinsic rotation
            t - 3 by 1 array, camera extrinsic translation
            imSize=None - a list of two numbers, image size [width, height]
            id=0 - integer, camera id
            dist=None - a list of five numbers, distortion factor.
        """
        self.id = id
        self.K = K
        self.R = R
        self.t = t
        self.P_norm = np.concatenate([R, t], axis=1)
        self.P = np.matmul(K, self.P_norm) # camera projection matrix (world to image)
        self.T = np.vstack([self.P_norm, np.zeros((1, self.P_norm.shape[1]))])
        self.T[-1,-1] = 1
        self.K_inv = np.linalg.inv(self.K)      # inverse of intrisic for convenience
        self.C_world_inhomo =np.matmul( - R.T , t)  # camera center in the world coordinate using inhomogeneous representation
        self.imSize = imSize                    # image size
        self.dist = dist                        # distortion factor

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
    
    def pixel_to_ray_vec(self, pts):
        """ given a pixel, calculate a line that all points will be projected to the pixel
        through the camera
        Params:
            pts - 2 by n array, points in image coordinates
        Return:
            d - 3 by n array, direction vector for each pixel
            C - 3 by 1 array, camera center in world frame, pencil of point representation.
        """
        pts_homo = homogenize(pts)
        pts_norm = np.matmul( self.K_inv, pts_homo)
        d = np.matmul(self.R.T, pts_norm)
        d = d / np.sign(d[0,:]) / np.linalg.norm(d, axis=0)
        C = self.C_world_inhomo
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
def test_get_image_coordinate():
    cam = camera_setup_6()
    X = cam.C_world_inhomo + 10
    x = cam.get_image_coordinate(X)
    print(x)

def test_pixel_to_ray(cam):
    x = np.array([[1, 500, 1000],
              [1, 200, 400]])
    for i in range(x.shape[1]):
        d, C = cam.pixel_to_ray(x[0,i], x[1,i])

def camera_setup_1():
    """ return the avl camera1 object """
    K = np.array([[1826.998004,    0.000000, 1174.548672],
                  [0.000000,    1802.603136,  776.028597],
                  [0.000000,       0.000000,    1.000000]])
    
    Rt = np.array([[ 1.5426360183850896e-01, -6.8597082105982421e-02,  9.8564556584725482e-01,  4.7539938241243362e-02],
                   [-9.8802970661938061e-01, -1.0912135033489312e-02,  1.5387730224640517e-01,  3.1389930844306946e-01],
                   [ 1.9996357324159053e-04, -9.9758476614047986e-01, -6.9459300162133530e-02, -5.5608768016099930e-02],
                   [0., 0., 0., 1. ]])
    R = Rt[0:3, 0:3].T
    t = -np.matmul(R, Rt[0:3, 3:4])

    imSize = [1920, 1440]
    dist = np.array([-0.136981, 0.043159, 0.006235, 0.018954, 0.000000])
    cam = Camera(K, R, t, imSize=imSize, id=1, dist=dist)
    return cam

def camera_setup_6():
    """ return the avl camera6 object """
    K = np.array([[1790.634474, 0., 973.099292],
                  [0., 1785.950534, 803.294457],
                  [0., 0., 1. ]])
    
    Rt = np.array([[ -2.1022535018250471e-01, -9.2112145235168197e-02, 9.7330398891652492e-01, -1.4076865278184414e-02],
                   [ -9.7735897207277012e-01, -4.6117027185500481e-03, -2.1153763709301088e-01, -3.1732881069183350e-01],
                   [ 2.3973774202277975e-02, -9.9573795995643932e-01, -8.9057134763516621e-02, -7.2184838354587555e-02],
                   [ 0., 0., 0., 1. ]])
    R = Rt[0:3, 0:3].T
    t = -np.matmul(R, Rt[0:3, 3:4])

    imSize = [1920, 1440]
    dist = np.array([-0.191070, 0.100324, 0.004250, -0.003317, 0.000000])
    cam = Camera(K, R, t, imSize=imSize, id=6, dist=dist)
    return cam

def test_camera_setup():
    cam = camera_setup_6()
    U, D, VT = np.linalg.svd(cam.P)
    C_homo = VT.T[:,-1::]
    C = C_homo[0:3] / C_homo[3,0]
    print(np.matmul(cam.P, C_homo).T)
    print("camera center:", C.T)
    w, theta = parameterize_rotation(cam.R)
    print(w.T, theta)
    pass

    
def test_pixel_to_ray_plot():
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
# main

def main():
    test_camera_setup()
    test_get_image_coordinate()


if __name__ == "__main__":
    main()
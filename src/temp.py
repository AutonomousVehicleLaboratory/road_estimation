

from camera import Camera
from plane_3d import Plane3D

def homogenize(pts):
    return np.vstack([pts, np.ones((1, pts.shape[1]))])

def pixel_to_ray(self, pts):
    pts_homo = homogenize(pts)
    pts_norm = np.matmul( self.K_inv, pts_homo)
    d = np.matmul(self.R.T, pts_norm)
    C = self.C_world_inhomo
    return d, C

def test_image_to_world():
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
    pts = np.array([[1000, 1300], 
                    [1000, 1000]])
    
    plane = Plane3D(0., 0., 1, 2)
    for i in range(x.shape[1]):
        d, C = pixel_to_ray(cam, pts)
        intersection = plane.plane_ray_intersection(d, C)
        print(x[0,i],'\t', x[1,i],'\t', intersection[0,0],'\t', intersection[1,0],'\t', intersection[2,0])


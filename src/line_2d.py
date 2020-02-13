""" Line in 2D

Author: Henry Zhang
Date:January 29, 2020
"""

# module
import numpy as np
from matplotlib import pyplot as plt

# parameters


# classes
class Line2D:
    def __init__(self, a=0., b=0., c=0.):
        self.a = a
        self.b = b
        self.c = c
        self.normalize()
    
    @classmethod
    def fit(cls, data, method="least_square", weight=None):
        if method == "min":
            if len(data) == 2 and not np.all(data[0,:]-data[1,:]==0):
                a = data[0,1] - data[1,1]
                b = data[1,0] - data[0,0]
                c = -b*data[0,1] -a*data[0,0]
            else:
                print("Error: Incorrect data, two required for minimal model fitting.")
                exit()
        elif method == "least_square":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return cls(a,b,c)
    
    def eval(self, data):
        distance = np.abs(np.matmul(data , self.param[0:2,:]) + self.c).reshape([-1]) / np.sqrt(self.a**2 + self.b**2)
        return distance

    def normalize(self):
        s = np.sqrt(self.a**2 + self.b**2)
        if s == 0:
            print("Error: a = 0 and b = 0!!!")
            exit()
        else:
            self.a, self.b, self.c = self.a/s, self.b/s, self.c/s
            self.param = np.array([[self.a, self.b, self.c]]).T
    
    def vis(self, ax, lim=[0, 100], linewidth=2):
        if not self.b == 0:
            y = -self.a * np.array(lim) / self.b - self.c / self.b
            ax.plot(lim, y, linewidth=linewidth)



# functions
def test_line_2d():
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    data = np.array([[2,3], [40,40]])
    line2 = Line2D.fit(data, method="min")
    ax.scatter(data[:,0], data[:,1])
    line2.vis(ax)

    plt.show()

def test_outlier():
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    outlier_num = 40
    data = np.random.randn(100, 2)/10 + np.arange(0,100,1).reshape([-1,1])
    data[np.random.randint(0, data.shape[0], outlier_num), 1] += np.random.randint(-50, 50, outlier_num)
    
    line = Line2D(-1,1,-1)
    threshold = 1
    cost_data = line.eval(data)
    inlier = data[cost_data <= threshold**2, :]
    outlier = data[cost_data > threshold**2, :]
    line.vis(ax)
    ax.scatter(inlier[:,0], inlier[:,1])
    ax.scatter(outlier[:,0], outlier[:,1])
    # ax.scatter(data[:,0], data[:,1])
    plt.show()

# main
def main():
    test_line_2d()
    test_outlier()

if __name__ == "__main__":
    main()
""" Random Sample Consensus (RANSAC) and its variations

Author: Henry Zhang
Date:January 29, 2020
"""

# module
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from line_2d import Line2D
from plane_3d import Plane3D
from copy import deepcopy

# parameters


# classes

# functions
class RANSAC:
    def __init__(self, model, min_sample_num, threshold, iteration=500, method="MSAC"):
        self.model = model
        self.min_sample_num = min_sample_num
        self.inlier = None
        self.outlier = None
        self.cost = None
        self.threshold = threshold
        self.iteration = iteration
        self.method = method

    def ransac(self, data, ax=None, weight = None, constraint = None, constraint_threshold = None):
        ctr = 0
        cost_best = np.inf
        model_best = None
        sample_num = self.min_sample_num
        while ctr < self.iteration:
            idx = self.non_repeat_index(0, data.shape[0], sample_num)
            subset = data[idx, :]
            model = self.model.fit(subset, method="min", weight=weight)
            if constraint is None:
                cost = self.eval(model, data)
            elif self.model.satisfy_constraint(model, constraint, constraint_threshold):
                cost = self.eval(model, data)
            else:
                cost = np.inf
            if cost < cost_best:
                model_best = deepcopy(model) # BUG NOTE: this mistake happens again!!!
                cost_best = cost
                if not ax is None:
                    model_best.vis(ax)
                    # ax.scatter(self.inlier[:,0], self.inlier[:,1])
                    # ax.scatter(self.outlier[:,0], self.outlier[:,1])
                    plt.pause(1)
                    print(cost_best, cost, len(self.inlier), len(self.outlier))
            # jump out condition
            ctr += 1
        cost_best = self.eval(model_best, data)
        self.model.satisfy_constraint(model_best, constraint, constraint_threshold, debug=True)
        return model_best, cost_best, self.inlier, self.outlier
    
    def eval(self, model, data):
        cost_data = model.eval(data)
        cost_data[cost_data>self.threshold**2] = self.threshold**2
        self.inlier = data[cost_data < self.threshold**2, :]
        self.outlier = data[cost_data >= self.threshold**2, :]
        if self.method == "MSAC":
            cost = np.sum(cost_data)
        elif self.method == "RANSAC":
            cost = np.sum(cost_data>=self.threshold)*self.threshold**2
        else:
            print("no such method in eval")
            exit()
        self.cost = cost
        return cost
    
    def non_repeat_index(self, low, high, num):
        idx = []
        if num > high - low:
            print("Error: impossible to generate non repeat when pool is smaller than sample!")
            exit()
        else:
            while len(idx) < num:
                index = np.random.randint(low, high)
                if not index in idx:
                    idx.append(index)
        return idx

def test_ransac_2d():
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    outlier_num = 40
    data = np.random.randn(100, 2)/10 + np.arange(0,100,1).reshape([-1,1])
    data[np.random.randint(0, data.shape[0], outlier_num), 1] += np.random.randint(-50, 50, outlier_num)
    
    # ax.scatter(data[:,0], data[:,1])
    
    sac = RANSAC(Line2D, min_sample_num=2, threshold=1., iteration=500, method="MSAC")
    line_best, cost_best, inlier, outlier = sac.ransac(data, ax=ax)
    line_best.vis(ax, linewidth=3)
    ax.scatter(inlier[:,0], inlier[:,1])
    ax.scatter(outlier[:,0], outlier[:,1])
    print(line_best.param.T, cost_best)
    
    plt.show()

def test_ransac_3d():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    
    a_true, b_true, c_true, d_true = 1, 1, -1, 1
    outlier_num = 40
    data = np.zeros((100, 3))
    data[:, 0] = np.repeat(np.arange(0, 10, 1), 10)
    data[:, 1] = np.tile(np.arange(0, 10, 1), 10)
    data[:, 2] = (- a_true * data[:,0] - b_true * data[:,1] - d_true) / c_true
    data[np.random.randint(0, data.shape[0], outlier_num), 2] += np.random.randint(-50, 50, outlier_num)
    print(data[0:5, :].T)

    ax.scatter(data[:,0], data[:,1], data[:,2])

    sac = RANSAC(Plane3D, min_sample_num=3, threshold=1, iteration=50, method="MSAC")
    plane_best, cost_best, inlier, outlier = sac.ransac(data, ax=ax)
    plane_best.vis(ax)
    ax.scatter(inlier[:,0], inlier[:,1], inlier[:,2])
    ax.scatter(outlier[:,0], outlier[:,1], outlier[:,2])
    print(plane_best.param.T, cost_best)

    plt.show()

# main
def main():
    test_ransac_2d()
    test_ransac_3d()

if __name__ == "__main__":
    main()
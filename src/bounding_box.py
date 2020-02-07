""" Bounding box

Author: Henry Zhang
Date:February 05, 2020
"""

# module
import numpy as np

# parameters


# classes
class BoundingBox:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.bbox = np.array([left, top, right, bottom])
    
    def bottom_point(self):
        """ calculate the botton middle point of the bounding box, 
        
        Params:
            None
        Return:
            Ix, Iy - the x, y position of the pixel
        """
        return (self.left + self.right) / 2.0, self.bottom
    
    def vis(self, ax):
        x = [self.left, self.left, self.right, self.right, self.left]
        y = [self.top, self.bottom, self.bottom, self.top, self.top]
        ax.plot(x, y)

    
# functions


# main
def main():
    pass

if __name__ == "__main__":
    main()
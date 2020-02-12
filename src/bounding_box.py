""" Bounding box

Author: Henry Zhang
Date:February 05, 2020
"""

# module
import numpy as np

# parameters


# classes
class BoundingBox:
    def __init__(self, left, top, width, height, angle=0.0, label="None"):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.label = label
        self.angle = 0.0 # currently do not accept angle
        self.bbox = np.array([left, top, width, height])
    
    def bottom_point(self):
        """ calculate the botton middle point of the bounding box, 
        
        Params:
            None
        Return:
            Ix, Iy - the x, y position of the pixel
        """
        return self.left + (self.width / 2.0), self.top + self.height
    
    def vis(self, ax):
        x = [self.left, self.left, self.left+self.width, self.left+self.width, self.left]
        y = [self.top, self.top+self.height, self.top+self.height, self.top, self.top]
        ax.plot(x, y)
        ax.text(self.left, self.top, self.label)

    
# functions


# main
def main():
    pass

if __name__ == "__main__":
    main()
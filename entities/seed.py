import cv2 as cv
import numpy as np


def crop_bottom(image, amt_left=2):
    cropped_img = image[int(image.shape[0]/amt_left):image.shape[0]]
    return cropped_img


class SeedSection:
    """
    Represents a section of a seed in a image.
    """

    img = None
    root_img = None
    root_contours = []

    #debugging images
    root_mask = None

    def __init__(self, mat):
        self.img = mat
        self.root_img = crop_bottom(mat, 2.5)

        h, w = self.img.shape[:2]
        self.width = w
        self.height = h

        print('[SeedSection]: Creating with size(%d, %d)' % (self.width, self.height))
        self.find_root_contours()

    def find_root_contours(self):
        """
        Finds the contours of the roots in the section
        """
        # Convert the image to hsv.
        hsv = cv.cvtColor(self.root_img, cv.COLOR_BGR2HSV)
        
        # get the upper range of hsv values for roots
        upper_hsv = cv.inRange(hsv, np.array([100, 0, 140]), np.array([255, 30, 180]))
        
        # Threshold the upper_hsv image so that we can find contours
        _, thresh = cv.threshold(upper_hsv, 10, 255, cv.THRESH_BINARY)
        self.root_mask = thresh

        self.root_contours, _ = cv.findContours(
            thresh.copy(),
            cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE
        )

import cv2 as cv
import numpy as np
from entities.location import Location

def get_image_below_y(image, y):
    h, w = image.shape[:2]
    cropped_img = image[y : h, 0 : w]
    new_h, new_w = cropped_img.shape[:2]
    print('[get_image_below_y]: Cropped (%d, %d) to (%d, %d) using y=[%d]!' % (w, h, new_w, new_h, y))
    return cropped_img.copy()


class SeedSection:
    """
    Represents a section of a seed in a image.
    """

    img = None
    root_img = None
    root_contours = []

    #debugging images
    combined_hsv = None

    def __init__(self, mat, seed_centroid: Location):
        self.img = mat
        self.seed_centroid = seed_centroid
        self.root_img = get_image_below_y(mat, seed_centroid.y)

        h, w = self.img.shape[:2]
        self.width = w
        self.height = h

        print('[SeedSection]: Created with size(%d, %d)' % (self.width, self.height))
        self.__find_root_contours()

    def __find_root_contours(self):
        """
        Finds the contours of the roots in the section
        """
        # Convert the image to hsv.
        hsv = cv.cvtColor(self.root_img, cv.COLOR_BGR2HSV)
        
        # get the range of hsv values for roots
        upper_hsv = cv.inRange(hsv, np.array([0, 0, 150]), np.array([25, 20, 180]))
        # upper_hsv2 = cv.inRange(hsv, np.array([10, 10, 150]), np.array([25, 20, 190]))
        lower_hsv = cv.inRange(hsv, np.array([120, 0, 150]), np.array([155, 25, 170]))
        lower_hsv2 = cv.inRange(hsv, np.array([150, 5, 140]), np.array([180, 25, 165]))

        combined_hsv = cv.bitwise_or(upper_hsv, lower_hsv)
        combined_hsv = cv.bitwise_or(combined_hsv, lower_hsv2)
        # combined_hsv = cv.bitwise_or(combined_hsv, upper_hsv2)
        self.combined_hsv = combined_hsv
        
        # Threshold the upper_hsv image so that we can find contours
        _, thresh = cv.threshold(combined_hsv, 1, 255, cv.THRESH_BINARY)
        self.root_mask = thresh

        contours, _ = cv.findContours(
            thresh.copy(),
            cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            area: float = cv.contourArea(c)

            if 30 < area < 2000:
                self.root_contours.append(c)

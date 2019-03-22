import cv2 as cv
import numpy as np
from entities.roots import RootAngle, RootInfo
from entities.location import Location

def crop_bottom(image, amt_left=2):
    cropped_img = image[int(image.shape[0]/amt_left):image.shape[0]]
    return cropped_img

def find_centroid(contour) -> Location:
    m = cv.moments(contour)
    if (m == None or m['m00'] == 0):
        return Location(0, 0)

    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])
    return Location(cx, cy)

class RangleImage:
    """
    Represents the original image taken by the user, but contains additional attributes
    about the image's contents.
    """

    #debugging images
    seed_mask = None

    def __init__(self, mat):
        self.img = mat
        self.seeds = []
        self.seed_contours = []

    def findSeeds(self):
        hue_img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)

        lower_hue_range = cv.inRange(hue_img, np.array([0, 45, 140]), np.array([20, 95, 215]))
        upper_hue_range = cv.inRange(hue_img, np.array([10, 20, 160]), np.array([40, 50, 200]))

        lower_mask = cv.addWeighted(lower_hue_range, 1.0, 1, 1.0, 0.0)
        upper_mask = cv.addWeighted(upper_hue_range, 1.0, 1, 1.0, 0.0)

        seed_mask = cv.bitwise_or(lower_mask, upper_mask)
        self.seed_mask = seed_mask

        self.seed_contours = self.__findSeedContours(seed_mask)
    
    def __findSeedContours(self, seed_mask) -> list:
        # threshold the image
        _, threshed_img = cv.threshold(seed_mask, 1, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(
            threshed_img.copy(),
            cv.RETR_LIST,
            cv.CHAIN_APPROX_NONE
        )

        h, w, = self.img.shape[:2]
        
        filtered_contours = []

        # Filter contours to only be those with a certain area
        for c in contours:
            area: float = cv.contourArea(c)
            centroid = find_centroid(c)

            is_not_near_top = centroid.y > h / 2.25
            is_not_near_edge = centroid.x > (w / 12) and centroid.x < (w - (w / 12))
            has_area_of_a_seed = (2000 < area < (h * w / 8))

            if is_not_near_top and is_not_near_edge and has_area_of_a_seed:
                filtered_contours.append(c)

        return filtered_contours

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
        self.findRootContours()

    def findRootContours(self):
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


class RootFinder:
    """
    Finds the root in a SeedSection
    """
    @staticmethod
    def find(section: SeedSection) -> RootAngle:
        """
        Finds information about the roots in the section
        Parameters
        ----------
        section : SeedSection
            The seed section containing the roots we want to find.
        """
        left_root = RootInfo()
        right_root = RootInfo()
        return RootAngle(left_root, right_root)


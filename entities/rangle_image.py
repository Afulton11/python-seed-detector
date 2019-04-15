import cv2 as cv
import numpy as np
from typing import List
from entities.location import Location
from entities.seed import SeedSection
from entities.utils import find_centroid, clamp


class RangleImage:
    """
    Represents the original image taken by the user, but contains additional attributes
    about the image's contents.
    """

    #debugging images
    seed_mask = None

    def __init__(self, mat):
        self.original_image = mat
        
        blur = cv.medianBlur(mat.copy(), 21)
        # use a bilateral blur to keep edges sharp, but blend similar colors
        self.blurred_image = cv.bilateralFilter(blur, 9, 75, 75)

        self.seeds = List[SeedSection]
        self.seed_contours = []

        h, w = self.original_image.shape[:2]
        self.original_width = w
        self.original_height = h

    def findSeeds(self):
        hue_img = cv.cvtColor(self.blurred_image, cv.COLOR_BGR2HSV)

        #
        # Apply Another blur to hsv image, We only want the large blobs that stand out
        # with specific hsv values.
        #
        blurred_hue = cv.medianBlur(hue_img.copy(), 25)
        # use a bilateral blur to keep edges sharp, but blend similar colors
        blurred_hue = cv.bilateralFilter(blurred_hue, 7, 201, 201)

        self.hsv_img = blurred_hue.copy()

        lower_hue_range = cv.inRange(blurred_hue, np.array([0, 45, 140]), np.array([20, 95, 215]))
        upper_hue_range = cv.inRange(blurred_hue, np.array([10, 20, 160]), np.array([40, 50, 200]))

        lower_mask = cv.addWeighted(lower_hue_range, 1.0, 1, 1.0, 0.0)
        upper_mask = cv.addWeighted(upper_hue_range, 1.0, 1, 1.0, 0.0)

        seed_mask = cv.bitwise_or(lower_mask, upper_mask)

        # connect close blobs
        seed_mask = cv.dilate(seed_mask, np.ones((10, 10), np.uint8), iterations=5)

        # Remove random blobs
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
        seed_mask = cv.morphologyEx(seed_mask, cv.MORPH_OPEN, kernel)

        self.seed_mask = seed_mask

        self.seed_contours = self.__find_seed_contours(seed_mask)
        self.seeds = self.__create_seed_sections()


    def __find_seed_contours(self, seed_mask) -> list:
        # threshold the image
        _, threshed_img = cv.threshold(seed_mask, 1, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(
            threshed_img.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE
        )
        
        w = self.original_width
        h = self.original_height

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

    def __create_seed_sections(self) -> List[SeedSection]:
        sections = []
        seed_width: int = self.get_seed_image_width()
        top: int = 0
        bottom: int = self.get_seed_image_height()
        for c in self.seed_contours:
            centroid: Location = find_centroid(c)
            left: int = int(clamp(centroid.x - (seed_width / 2), 0, self.original_width))
            right: int = int(clamp(centroid.x + (seed_width / 2), 0, self.original_width))

            seed_image = self.blurred_image[top : bottom, left : right]
            relative_seed_centroid_x = centroid.x - left

            _, ext_bot_y = tuple(c[c[:, :, 1].argmax()][0])
            
            section = SeedSection(seed_image, relative_seed_centroid_x, ext_bot_y - 100)
            sections.append(section)

        return sections

    def get_seed_image_width(self) -> int:
        return int(self.original_width / 6)


    def get_seed_image_height(self) -> int:
        return self.original_height
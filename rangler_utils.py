from enum import Enum
import cv2 as cv
# import numpy as np


def find_and_draw_contours(bgr_img, contour_area_predicate, thresh_value=10):
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray_img, thresh_value, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    rangler_image = RanglerImage(bgr_img);

    for c in contours:
        area = cv.contourArea(c)
        if contour_area_predicate(area):
            rangler_image.add_contour(c)

    rangler_image.draw_contours()

    return rangler_image.img


class RanglerImage:
    """
    Represents a Mat in opencv
    """
    def __init__(self, mat):
        self.img = mat
        self.__areas = []
        self.__contours = []

    def combine(self, other):
        """
        Combines this image and the other image
        using the opencv bitwise OR. This also combines their areas.
        :param other: the other image to combine
        :return: the combined Image
        """
        combined_mat = cv.bitwise_or(self.img, other.img)
        combined_img = RanglerImage(combined_mat)
        combined_img.__areas = self.__areas + other.__areas
        combined_img.__contours = self.__contours + other.__contours
        return combined_img

    def add_area(self, area):
        self.__areas.append(area)

    def add_contour(self, contour):
        self.__contours.append(contour)

    def get_first_area(self, predicate):
        for a in self.__areas:
            if predicate(a):
                return a

    def draw_areas(self):
        for a in self.__areas:
            a.draw(self.img)

    def draw_contours(self):
        length = len(self.__contours)
        for index in range(length):
            cv.drawContours(self.img, self.__contours, index, (0, 255, 0), thickness=3)


class LocationType(Enum):
    SEED = 0,
    ROOT = 1,
    OTHER = 2,


class ImageLocation:
    """
    Represents a location in an image
    """
    def __init__(self, kind, x, y, width, height):
        """

        :param type: LocationType (i.e. LocationType.SEED)
        :param x: the x position
        :param y: the y position
        :param width: the width of the location
        :param height: the height of the location
        :return: ImageLocation
        """
        self.kind = kind
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __upper_left(self):
        return int(self.x - self.width / 2.0), int(self.y - self.height / 2.0)

    def __bottom_right(self):
        return int(self.x + self.width / 2.0), int(self.y + self.height / 2.0)

    def get_img_rect(self, img):
        upper_left = self.__upper_left()
        bottom_right = self.__bottom_right()
        return self.img[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

    def draw(self, img):
        upper_left = self.__upper_left()
        bottom_right = self.__bottom_right()
        cv.rectangle(
            img,
            upper_left,
            bottom_right,
            (255, 255, 0),
            thickness=3
        )

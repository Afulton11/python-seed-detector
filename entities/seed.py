import sys
import math
import cv2 as cv
import numpy as np
from entities.utils import find_centroid, clamp
from entities.location import Location

class Line:
    """
    Represents a openCv line
    """
    def __init__(self, point1: Location, point2: Location):
        assert(hasattr(point1, 'x'))
        assert(hasattr(point1, 'y'))
        assert(hasattr(point2, 'x'))
        assert(hasattr(point2, 'y'))
        self.point1 = point1
        self.point2 = point2

    def _slope(self):
        vx = float(self.point1.x - self.point2.x)
        vy = float(self.point1.y - self.point2.y)

        if vx == 0:
            return sys.maxsize

        return vy / vx

    def _b(self):
        return (self.point1.y - self._slope() * self.point1.x)

    def distance(self, point: Location) -> int:
        m = self._slope()
        b = self._b()
        if m == sys.maxsize:
            return sys.maxsize
    
        line_point = Location(point.x, m * point.x + b)
        dist = line_point.distance(point)

        print('[distance]: point=[%s] line_pint=[%s] dist=[%d]' % (point, line_point, dist))
        return int(round(dist))

    def combine(self, other):
        def avgValue(x_1, x_2) -> int:
            return int(round((x_1 + x_2) / 2))
        def avgPoint(point_1, point_2) -> Location:
            return Location(avgValue(point_1.x, point_2.x), avgValue(point_1.y, point_2.y))
        
        return Line(
            avgPoint(self.point1, other.point1),
            avgPoint(self.point2, other.point2)
        )

    def draw(self, mat):
        print('[Drawing line]: %s' % (self.__str__()))
        cv.line(mat, (self.point1.x, self.point1.y), (self.point2.x, self.point2.y), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)

    def draw_entire_width(self, mat):
        print('[Drawing line]: %s' % (self.__str__()))
        h, w = mat.shape[:2]
        m = self._slope()
        b = self._b()

        left_point = Location(-1, -1)
        right_point = Location(-1, -1)

        left_point.x = 0
        left_point.y = int(round(b))

        right_point.x = w
        right_point.y = int(round(m * right_point.x + b))

        cv.line(mat, (left_point.x, left_point.y), (right_point.x, right_point.y), (0, 0, 255), thickness=2, lineType=cv.LINE_AA)

    def __str__(self):
        return '%s, %s, m=%d' % (self.point1, self.point2, self._slope())


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

    def __init__(self, mat, seed_centroid: Location, relative_seed_x: int):
        self.img = mat
        self.seed_center_x = relative_seed_x
        self.root_img = get_image_below_y(mat, seed_centroid.y)
        self.root_contours = []
        self.root_lines = []

        h, w = self.img.shape[:2]
        self.width = w
        self.height = h

        print('[SeedSection]: Created with size(%d, %d)' % (self.width, self.height))
        self.__find_root_contours()
        self.__find_lines()

    def __find_root_contours(self):
        """
        Finds the contours of the roots in the section
        """
        # Convert the image to hsv.
        hsv = cv.cvtColor(self.root_img, cv.COLOR_BGR2HSV)
        
        # get the range of hsv values for roots
        upper_hsv = cv.inRange(hsv, np.array([0, 0, 150]), np.array([25, 20, 180]))
        # upper_hsv2 = cv.inRange(hsv, np.array([10, 10, 150]), np.array([25, 20, 190]))
        lower_hsv = cv.inRange(hsv, np.array([120, 0, 130]), np.array([155, 25, 170]))
        lower_hsv2 = cv.inRange(hsv, np.array([150, 5, 140]), np.array([180, 25, 165]))

        combined_hsv = cv.bitwise_or(upper_hsv, lower_hsv)
        combined_hsv = cv.bitwise_or(combined_hsv, lower_hsv2)

        # Remove noise
        remove_noise_kernel = np.ones((10, 10), np.uint8)
        combined_hsv = cv.morphologyEx(combined_hsv, cv.MORPH_OPEN, remove_noise_kernel)

        # connect close blobs
        threshed = cv.dilate(combined_hsv, np.ones((2, 2), np.uint8), iterations=10)

        # debug combined image
        self.combined_hsv = combined_hsv
        
        # Threshold the upper_hsv image so that we can find contours
        # _, thresh = cv.threshold(threshed, 1, 255, cv.THRESH_BINARY)
        self.root_mask = threshed

        contours, _ = cv.findContours(
            threshed.copy(),
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            # hull = cv.convexHull(c)
            hull = c
            area: float = cv.contourArea(hull)

            if 100 < area < (self.height * self.width / 2):
                self.root_contours.append(hull)

    def __find_lines(self):
        for c in self.root_contours:
            # determine the most extreme points along the contour
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            point_1 = Location(-1, -1)
            point_2 = Location(-1, -1)

            cv.circle(self.root_img, extLeft, 8, (0, 0, 255), -1)
            cv.circle(self.root_img, extRight, 8, (0, 255, 0), -1)
            cv.circle(self.root_img, extTop, 8, (255, 0, 0), -1)
            cv.circle(self.root_img, extBot, 8, (255, 255, 0), -1)
            
            point_1 = Location(extTop[0], extTop[1])
            point_2 = Location(extBot[0], extBot[1])

            line = Line(point_1, point_2)

            self.root_lines.append(line)

        # loop through one more time, combining all similar lines.
        lines_copy = self.root_lines.copy()
        for i, line in enumerate(lines_copy):
            for j, other_line in enumerate(lines_copy):
                if (j == i):
                    continue
                dist = line.distance(other_line.point1)
                v_slope = abs(other_line._slope() - line._slope())
                print('[Slope_diff]: %lf' % (v_slope))
                if v_slope < 1.5 and dist < self.width / 3:
                    if (line in self.root_lines):
                        self.root_lines.remove(line)
                    if (other_line in self.root_lines):
                        self.root_lines.remove(other_line)

                    line = line.combine(other_line)
                    self.root_lines.append(line)

    def draw_lines(self, mat):
        seed_point = Location(self.seed_center_x, 0)
        for line in self.root_lines:
            dist = line.distance(seed_point)
            if dist < self.width:
                line.draw_entire_width(mat)

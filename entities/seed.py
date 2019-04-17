import math
import cv2 as cv
import numpy as np
from entities.utils import find_centroid, clamp, get_cv_max_int
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
        self.dx = float(self.point1.x - self.point2.x)
        self.dy =float(self.point1.y - self.point2.y)

    def _slope(self):
        try:
            return self.dy / self.dx
        except ZeroDivisionError:
            return get_cv_max_int()

    def _b(self):
        return (self.point1.y - self._slope() * self.point1.x)

    def distance(self, point: Location) -> float:
        m = self._slope()
        b = self._b()

        line_point = self.pointAtX(point.x)
        dist = line_point.distance(point)

        return dist

    def average(self, other):
        return Line(
            self.point1.average(other.point1),
            self.point2.average(other.point2)
        )
    
    def pointAtX(self, x: float) -> Location:
        y =  self._slope() * x + self._b()
        if (self._slope() >= get_cv_max_int()):
            y = x
        return Location(x, y)

    def pointAtY(self, y: float) -> Location:
        x =  (y / self._slope()) - self._b()
        if (self._slope() >= get_cv_max_int()):
            x = y
        return Location(x, y)

    def angleWithHorizontal(self) -> float:
        """
        returns the angle (in radians) this line makes with any horizontal line
        """
        return math.atan2(self.dy, self.dx)

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
        left_point.y = b

        right_point.x = w
        right_point.y = m * right_point.x + b

        cv.line(mat, tuple(left_point), tuple(right_point), (0, 0, 255), thickness=2)

    def __str__(self):
        return '%s, %s, m=%d' % (self.point1, self.point2, self._slope())


class ContourInfo:
    """
    A class that contains a line created from the contour's extreme top and bottom points.
    Also a line drawn from a seed's center, to the averaged extreme point of the contour.
    """

    def __init__(self, contour_line: Line, seed_line: Line):
        self.contour_line = contour_line
        self.seed_line = seed_line

    def average_contour(self, other):
        self.contour_line = self.contour_line.average(other.contour_line)
        # self.seed_line = self.seed_line.average(other.seed_line)
        return self

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

    def __init__(self, mat, seed_x: float, seed_y: float):
        self.img = mat
        self.seed_center_x = seed_x
        self.root_img = get_image_below_y(mat, seed_y)
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
        
        #
        # Apply Another blur to hsv image, We only want the large blobs that stand out
        # with specific hsv values.
        #
        blurred_hue = cv.medianBlur(hsv, 21)
        # use a bilateral blur to keep edges sharp, but blend similar colors
        hsv = cv.bilateralFilter(blurred_hue, 9, 75, 75)

        # get the range of hsv values for roots
        upper_hsv = cv.inRange(hsv, np.array([130, 0, 150]), np.array([160, 15, 180]))
        lower_hsv = cv.inRange(hsv, np.array([100, 0, 110]), np.array([130, 15, 170]))

        combined_hsv = cv.bitwise_or(upper_hsv, lower_hsv)
        self.combined_hsv = combined_hsv.copy()

        # Remove noise
        remove_noise_kernel = np.ones((11, 11), np.uint8)
        combined_hsv = cv.morphologyEx(combined_hsv, cv.MORPH_OPEN, remove_noise_kernel)

        # connect close blobs
        threshed = cv.dilate(combined_hsv, np.ones((3, 3), np.uint8), iterations=5)

        # debug combined image
        # self.combined_hsv = combined_hsv
        
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
        seed_point = Location(self.seed_center_x, 0)
        tmp_contour_infos = []
        for c in self.root_contours:
            # determine the most extreme points along the contour
            extLeft = Location.from_tuple(tuple(c[c[:, :, 0].argmin()][0]))
            extRight = Location.from_tuple(tuple(c[c[:, :, 0].argmax()][0]))
            extTop = Location.from_tuple(tuple(c[c[:, :, 1].argmin()][0]))
            extBot = Location.from_tuple(tuple(c[c[:, :, 1].argmax()][0]))

            # cv.circle(self.root_img, tuple(extLeft), 8, (0, 0, 255), -1)
            # cv.circle(self.root_img, tuple(extRight), 8, (0, 255, 0), -1)
            # cv.circle(self.root_img, tuple(extTop), 8, (255, 0, 0), -1)
            # cv.circle(self.root_img, tuple(extBot), 8, (255, 255, 0), -1)
            
            averaged_extremes = extLeft.average(extRight).average(extTop).average(extBot)

            cv.circle(self.root_img, tuple(averaged_extremes), 8, (0, 0, 0), -1)

            seed_line = Line(seed_point, averaged_extremes)
            contour_line = Line(extTop, extBot)

            # cv.line(self.root_img, tuple(extTop), tuple(extBot), (200, 200, 200), thickness=3)

            if (contour_line.distance(seed_point) < self.height):
                contour_info = ContourInfo(contour_line, seed_line)
                tmp_contour_infos.append(contour_info)

        tmp_combined_infos = []

        for info in tmp_contour_infos:
            line: Line = info.contour_line
            matched: bool = False
            for other_info in tmp_contour_infos:
                other_line: Line = other_info.contour_line
                
                dist = line.distance(other_line.point1)
                v_slope = abs(other_line._slope() - line._slope())
                print('[Slope_diff]: %lf' % (v_slope))
                if v_slope < (self.height / self.width) and dist < self.width / 3:
                    matched = True
                    tmp_combined_infos.append(info.average_contour(other_info))
            
            if (not matched):
                tmp_combined_infos.append(info)

        for info in tmp_combined_infos:
            contour_line = info.contour_line

            distance = contour_line.distance(seed_point)
            if distance < self.width:
                self.root_lines.append(info.seed_line)


    def draw_lines(self, mat):
        for line in self.root_lines:
            line.draw_entire_width(mat)

    def get_rangle(self):
        largest_angle = 0
        for l in self.root_lines:
            angle = l.angleWithHorizontal()
            for l_other in self.root_lines:
                other_angle = l_other.angleWithHorizontal()
                if l != l_other and angle - other_angle > largest_angle:
                    largest_angle = angle - other_angle

        degrees = math.degrees(largest_angle)
        print('[get_rangle]: Found angle of : %lf' % (degrees))
        return largest_angle



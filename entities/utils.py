import cv2 as cv
from entities.location import Location


def find_centroid(contour) -> Location:
    m = cv.moments(contour)
    if (m == None or m['m00'] == 0):
        return Location(0, 0)

    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])
    return Location(cx, cy)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))

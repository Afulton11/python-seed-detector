import cv2 as cv
import numpy as np
from rangler_utils import RanglerImage, ImageLocation, LocationType


def get_root_mask(hue_img):
    root_upper_range = cv.inRange(hue_img, np.array([100, 0, 140]), np.array([255, 30, 180]))
    root_mask = cv.addWeighted(root_upper_range, 1.0, 1, 1.0, 0.0)

    return root_mask


def get_seed_mask(hue_img):
    lower_hue_range = cv.inRange(hue_img, np.array([0, 0, 140]), np.array([20, 100, 220]))
    upper_hue_range = cv.inRange(hue_img, np.array([0, 0, 150]), np.array([30, 50, 255]))

    lower_mask = cv.addWeighted(lower_hue_range, 1.0, 1, 1.0, 0.0)
    upper_mask = cv.addWeighted(upper_hue_range, 1.0, 1, 1.0, 0.0)

    seed_mask = cv.bitwise_or(lower_mask, upper_mask)

    return seed_mask

def find_blobs(grey_img, mask, blob_kind = LocationType.OTHER):
    blob_img = cv.bitwise_and(mask, grey_img)
    thresh, img_thresh = cv.threshold(blob_img, 1, 255, cv.THRESH_BINARY)
    # p_loc_bw_img = cv.adaptiveThreshold(p_loc_img,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,7,-3)

    blob_params = cv.SimpleBlobDetector_Params()
    blob_params.filterByConvexity = False

    blob_detector = cv.SimpleBlobDetector_create(blob_params)

    keypoints = blob_detector.detect(img_thresh)

    points = np.asarray([kp.pt for kp in keypoints])

    print(points)

    x_values = points[:, 0]
    y_values = points[:, 1]

    left_x = min(x_values)
    right_x = max(x_values)
    top_y = min(y_values)
    bottom_y = max(y_values)
    mid_x = left_x + (right_x - left_x) / 2.0
    mid_y = top_y + (bottom_y - top_y) / 2.0

    min_dist = -1
    mid_point = None

    bw_img_color = cv.cvtColor(img_thresh, cv.COLOR_GRAY2BGR)

    bw_img = RanglerImage(bw_img_color)

    # Note to self:
    # Our current algorithm is just finding the seed closest to the center.
    # This works for most images; however, for many images the wanted seed is off-center.
    # We can't use The numbers above the seeds as a reference point either because
    # the number isn't always shown clearly.
    # Maybe we could look for the number and if we find it, use the seed beneath that number
    # Otherwise, prompt the user to select a seed for the given image.

    for kp in keypoints:
        x = kp.pt[0]
        y = kp.pt[1]
        half_size = 30
        area = ImageLocation(blob_kind, x, y, half_size, half_size)
        bw_img.add_area(area)
        dist = abs(x - mid_x)
        dist_y = abs(y - mid_y)
        print("min_dist=%.2lf, this_dist=%.2lf, mid=(%.2lf, %.2lf), pt=(%.2lf, %.2lf), dist_y=%.2lf" % (
        min_dist, dist, mid_x, mid_y, x, y, dist_y))
        if dist_y < 250 and (dist < min_dist or min_dist < 0):
            print("\tsetting new dist!")
            min_dist = dist
            mid_point = kp.pt

    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    h, w = bw_img.img.shape[:2]

    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv.contourArea(con)
        if 3000 < area < (h * w) / 2:
            contours_area.append(con)

    cv.drawContours(bw_img.img, contours_area, -1, (0, 0, 255), thickness=3)

    bw_img.draw_areas();

    focus_x = int(mid_point[0])
    focus_y = int(mid_point[1])
    half_focus_area_width = 500 / 2
    half_focus_area_height = 600 / 2

    print(focus_x, focus_y)

    focus_img = cv.rectangle(
        bw_img.img,
        (int(focus_x - half_focus_area_width), int(focus_y + half_focus_area_width)),
        (int(focus_x + half_focus_area_width), int(focus_y - half_focus_area_height)),
        (0, 255, 0),
        thickness=6)

    return focus_img


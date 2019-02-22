import cv2 as cv
import numpy as np

img_path = '/Users/andrewfulton/Documents/School/Research/rangle/Root angle images/Bread Wheat/Images/IMG_7104.JPG'

# np.set_printoptions(threshold=np.nan)


def onMouseClicked(event, x, y, flags, img_param):
    if event == cv.EVENT_LBUTTONUP:
        print(img_param[y, x])


def show_image(name, img, w, h):
    screen_res = 1280.0, 720.0
    scale_width = screen_res[0] / w
    scale_height = screen_res[1] / h
    scale = min(scale_width, scale_height)
    window_width = int(w * scale)
    window_height = int(h * scale)

    print('showing image: %s (%d, %d)' % (name, window_width, window_height))
    img_resized = cv.resize(img, (window_width, window_height))
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, window_width, window_height)
    cv.imshow(name, img_resized)
    cv.setMouseCallback(name, onMouseClicked, img_resized)


def show_images(images, w, h):
    length = len(images)
    index = 0;
    while -1 < index < length:
        cur_img = images[index]
        show_image(cur_img[0], cur_img[1], w, h)
        key = cv.waitKey(0)
        cv.destroyWindow(cur_img[0])

        if key == 100:
            index += 1
        elif key == 27:
            break
        else:
            index -= 1


img = cv.imread(img_path, cv.IMREAD_COLOR)
h, w = img.shape[:2]

# get masks for planting locations
hue_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
lower_hue_range = cv.inRange(hue_img, np.array([10, 40, 140]), np.array([30, 70, 210]))
p_loc_mask = cv.addWeighted(lower_hue_range, 1.0, 1, 1.0, 0.0)

# Apply mask to black + white image
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
p_loc_img = cv.bitwise_and(p_loc_mask, grey_img)
thresh, p_loc_bw_img = cv.threshold(p_loc_img, 130, 255, cv.THRESH_BINARY)

blobDetectorParams = cv.SimpleBlobDetector_Params()
blobDetectorParams.blobColor = 255
blobDetectorParams.filterByConvexity = False

blobDetector = cv.SimpleBlobDetector_create(blobDetectorParams)

keypoints = blobDetector.detect(p_loc_bw_img)

points = np.asarray([kp.pt for kp in keypoints])

x_values = points[:, 0]
y_values = points[:, 1]

left_x = min(x_values)
right_x = max(x_values)
mid_x = left_x + (right_x - left_x) / 2.0

min_dist = -1
mid_key_index = 0

for i, kp in enumerate(keypoints):
    x = kp.pt[0]
    dist = abs(x - mid_x)
    if dist < min_dist or min_dist < 0:
        min_dist = dist
        mid_key_index = i


focus_x = int(x_values[mid_key_index])
focus_y = int(y_values[mid_key_index])
half_focus_area_width = 300 / 2
half_focus_area_height = 500 / 2

print(focus_x, focus_y)

p_loc_img_color = cv.cvtColor(p_loc_img, cv.COLOR_GRAY2BGR)
focus_img = cv.rectangle(
    p_loc_img_color,
    (focus_x - half_focus_area_width, focus_y + half_focus_area_width),
    (focus_x + half_focus_area_width, focus_y - half_focus_area_height),
    (0, 255, 0),
    thickness=6)

show_images([
    ('focus_rec', focus_img),
    ('p_loc_bw', p_loc_bw_img),
    ('p_loc', p_loc_img),
    ('original', img),
    ('hue_img', hue_img),
    ('lower_hue_range', lower_hue_range),
    ('lower_mask', p_loc_mask),
    ('grey', grey_img),
], w, h)

cv.destroyAllWindows()

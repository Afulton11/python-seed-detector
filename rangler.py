import cv2 as cv
import numpy as np

img_path = '/Users/andrewfulton/Documents/School/Research/rangle/Root angle images/Bread Wheat/Images/IMG_7103.JPG'


# np.set_printoptions(threshold=np.nan)

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
lower_hue_range = cv.inRange(hue_img, np.array([0, 40, 150]), np.array([20, 90, 230]))
upper_hue_range = cv.inRange(hue_img, np.array([20, 90, 230]), np.array([80, 255, 255]))
p_loc_mask = cv.addWeighted(lower_hue_range, 1.0, 1, 1.0, 0.0)

# Apply mask to black + white image
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
p_loc_img = cv.bitwise_and(p_loc_mask, grey_img)
thresh, p_loc_bw_img = cv.threshold(p_loc_img, 30, 255, cv.THRESH_BINARY)

show_images([
    ('original', img),
    ('hue_img', hue_img),
    ('lower_hue_range', lower_hue_range),
    ('lower_mask', p_loc_mask),
    ('upper_hue_range', upper_hue_range),
    ('grey', grey_img),
    ('p_loc', p_loc_img),
    ('p_loc_bw', thresh),
], w, h)

cv.destroyAllWindows()
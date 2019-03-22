import cv2 as cv
import numpy as np
from entities.seed import RangleImage

img_path_1 = '/Users/andrewfulton/Documents/School/Research/rangle/Root angle images/Bread Wheat/Images/IMG_7106.JPG'
img_path_2 = '/Users/andrewfulton/Documents/School/Research/rangle/Root angle images/Durum NAM/Images/IMG_3432  (1) .JPG'

img = cv.imread(img_path_1, cv.IMREAD_COLOR)
h, w = img.shape[:2]

# np.set_printoptions(threshold=np.nan)

def run_back_projection():
    roi = cv.imread('seed.jpg')
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    target = cv.imread(img_path_1)
    hsvt = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    # calculating object histogram
    roihist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # normalize histogram and apply backprojection
    cv.normalize(roihist, roihist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256], 1)
    # Now convolute with circular disc
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    cv.filter2D(dst, -1, disc, dst)
    # threshold and binary AND
    ret, thresh = cv.threshold(dst, 10, 255, 0)
    thresh = cv.merge((thresh, thresh, thresh))
    res = cv.bitwise_and(target, thresh)
    stackedRes = np.vstack((target, thresh, res))

    cv.imwrite('res.jpg', stackedRes)

    run(src=target)


def run(src=img):
    rangler: RangleImage = RangleImage(src)

    rangler.findSeeds()

    img_with_seed_contours = rangler.img.copy()
    cv.drawContours(
        img_with_seed_contours,
        rangler.seed_contours,
        -1,
        (255, 0, 0),
        thickness=3
    )

    show_images([
        ('seed_mask', rangler.seed_mask),
        ('seed_contours', img_with_seed_contours),
        ('original', rangler.img),
        ('hsv', cv.cvtColor(rangler.img, cv.COLOR_BGR2HSV))
    ], w, h)

    cv.destroyAllWindows()


def on_mouse_clicked(event, x, y, flags, img_param):
    if event == cv.EVENT_LBUTTONUP:
        print('position: ')
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
    cv.moveWindow(name, 0, 0)
    cv.imshow(name, img_resized)
    cv.setMouseCallback(name, on_mouse_clicked, img_resized)


def show_images(images, w, h):
    length = len(images)
    index = 0
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


# run()
run_back_projection()
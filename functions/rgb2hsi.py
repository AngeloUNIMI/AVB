import cv2


def rgb2hsi(rgb_img):

    img_hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    return img_hsv


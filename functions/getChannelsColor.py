import cv2
import os
import functions


def getChannelsColor(img):

    # HSV
    hsv = functions.rgb2hsi(img)
    h, s, i = cv2.split(hsv)
    # cv2.normalize(h, h, 0, 255, cv2.NORM_MINMAX)

    # RGB
    b, g, r = cv2.split(img)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(img_yuv)

    return h, s, i, r, g, b, y, u, v

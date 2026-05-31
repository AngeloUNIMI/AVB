import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions.getChannelsColor import getChannelsColor


def plot3d(img_rgb, mask, label):

    centroids = {}

    # HSI / RGB
    h, s, i, r, g, b, y, u, v = getChannelsColor(img_rgb)

    centroids['rgb'] = [cv2.mean(r, mask)[0], cv2.mean(g, mask)[0], cv2.mean(b, mask)[0]]
    centroids['hsi'] = [cv2.mean(h, mask)[0], cv2.mean(s, mask)[0], cv2.mean(i, mask)[0]]
    centroids['yuv'] = [cv2.mean(y, mask)[0], cv2.mean(u, mask)[0], cv2.mean(v, mask)[0]]

    if label.lower() == 'abnormal':
        color = 'r'
    else:
        color = 'b'

    return centroids, color

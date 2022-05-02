import cv2
import numpy as np


def whiteNorm(img, plotta, filename):

    # minImg = np.min(np.min(img))
    # img = img - minImg

    """
    minC = 1e6
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        ex_img = cv2.bitwise_and(v, mask)
        meanC = ex_img.sum() / (mask/255).sum()
        if meanC < minC:
            minC = meanC
            centerBackground = i

            mask3 = np.dstack([mask]*3)  # Make it 3 channel
            ex_img3 = cv2.bitwise_and(img, mask3)
            meanBackground = ex_img3.sum() / (mask3/255).sum()

    # normalize white
    img = img * (255 / meanBackground)
    img = np.clip(img, 0, 255)

    """

    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    imgNormWhite = np.uint8(img)

    if plotta:
        cv2.imwrite(filename, imgNormWhite)

    return imgNormWhite
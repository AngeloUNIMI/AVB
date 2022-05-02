import cv2
import util
import numpy as np


def morphProcMask(mask_chosen, plotta, filename):

    # mask_fill = util.flood_fill(mask_chosen)
    mask_fill = mask_chosen

    kernel_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    morph = cv2.morphologyEx(mask_fill, cv2.MORPH_CLOSE, kernel_15)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_15)

    # morph = util.flood_fill(morph)

    morph = util.getLargestCC(morph)
    mask_final = morph.astype(np.uint8)*255

    if plotta == True:
        cv2.imwrite(filename, mask_final)

    return mask_final

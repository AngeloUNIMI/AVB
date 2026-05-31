import cv2
import matplotlib.pyplot as plt
import numpy as np


def chooseClusterInteractive(img, posList, centers, labels):

    # init
    mask_chosen = []
    closestCenter = []
    closestCluster = []

    # print()

    posNp = posList

    # print(posNp)
    maxOverlap = -1
    maskPos = np.uint8(np.zeros(img.shape[0:2]))
    areaMezz = 30  # 30
    maskPos[posNp[1]-areaMezz:posNp[1]+areaMezz, posNp[0]-areaMezz:posNp[0]+areaMezz] = 255

    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)

        mask_comb = cv2.bitwise_and(maskPos, mask)
        sumComb = mask_comb.sum()

        cv2.imwrite('t.jpg', mask_comb)

        # if mask[posNp[1], posNp[0]] == 255:
        if sumComb > maxOverlap:

            maxOverlap = sumComb

            """
            titleStr = 'Corresponding segmentation (press ESC to continue)'

            mask3 = np.dstack([mask]*3)  # Make it 3 channel
            imgSegm = cv2.bitwise_and(img, mask3)

            final_frame = cv2.hconcat((img, mask3, imgSegm))

            cv2.namedWindow(titleStr)
            cv2.imshow(titleStr, final_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

            closestCluster = i
            closestCenter = c
            mask_chosen = mask

            # break

    return mask_chosen, closestCenter, closestCluster

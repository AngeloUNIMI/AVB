import cv2
import os
import numpy as np


def displayImg(imgNormWhite, mask3, img_mask, filenameImg, label, label_subTypes, dirFrames):

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org1 = (50, 50)
    org2 = (50, 100)
    org3 = (50, 150)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    titleStr = 'Result (press ANY key to continue)'
    textResult1 = filenameImg
    textResult2 = 'Label: {0} (Label w. subtypes: {1})'.format(label.lower(), label_subTypes.lower())
    # textResult3 = 'Output: {0}'.format(output.lower())
    final_frame = cv2.hconcat((imgNormWhite, mask3, img_mask))
    cv2.putText(final_frame, textResult1,  org1, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(final_frame, textResult2,  org2, font, fontScale, color, thickness, cv2.LINE_AA)
    # cv2.putText(final_frame, textResult3,  org3, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.namedWindow(titleStr)
    cv2.setWindowProperty(titleStr, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(titleStr, 100, 100)
    cv2.imshow(titleStr, final_frame)
    cv2.imwrite(os.path.join(dirFrames, filenameImg), final_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def updateImg(img, output, output_subTypes, filenameImg, dirFrames):

    img = np.uint8(img)

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org3 = (50, 150)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    titleStr = 'Result (press ANY key to continue)'
    textResult3 = 'Output: {0} (Output w. subtypes: {1})'.format(output.lower(), output_subTypes.lower())
    cv2.putText(img, textResult3, org3, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imwrite(os.path.join(dirFrames, filenameImg), img)


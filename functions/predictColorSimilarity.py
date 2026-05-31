import os
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression

from functions.getChannelsColor import getChannelsColor


def predictColorSimilarity(dirTypes, imgNormWhite, mask):

    filenameImgType_all = os.listdir(dirTypes)

    # create data structure features
    x_train = np.zeros((len(filenameImgType_all), 6))
    y_train = np.zeros((len(filenameImgType_all)), dtype=np.int)

    # extract features and labels
    for numImgType, filenameImgType in enumerate(filenameImgType_all):

        imgType = cv2.imread(os.path.join(dirTypes, filenameImgType))
        labelType = filenameImgType.split('_')[0]
        if labelType.lower() == 'abnormal':
            labelTypeNum = 0
        else:
            labelTypeNum = 1

        h, s, i, r, g, b, y, u, v = getChannelsColor(imgType)

        x_train[numImgType, :] = [cv2.mean(y)[0], cv2.mean(u)[0], cv2.mean(v)[0],
                                  cv2.mean(r)[0], cv2.mean(g)[0], cv2.mean(b)[0]]
        y_train[numImgType] = labelTypeNum

    # train linear class
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)

    # extract features from image
    h, s, i, r, g, b, y, u, v = getChannelsColor(imgNormWhite)
    x_test = np.zeros((1, 6))
    x_test[0, :] = [cv2.mean(y, mask)[0], cv2.mean(u, mask)[0], cv2.mean(v, mask)[0],
                    cv2.mean(r, mask)[0], cv2.mean(g, mask)[0], cv2.mean(b, mask)[0]]
    output_num = logreg.predict(x_test)[0]

    if output_num == 0:
        output = 'abnormal'
    else:
        output = 'normal'

    return output

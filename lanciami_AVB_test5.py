import numpy as np
import cv2
import os
# import pandas as pd
import openpyxl
from datetime import datetime
import copy
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from scipy.spatial import distance_matrix
import time

# functions
import functions
import util

# callback for ginput
posList = np.array([0, 0])
def getxy(event, x, y, flags, param):
    global posList, imgNormWhite, cache
    if event == cv2.EVENT_LBUTTONDOWN:
        posList[0] = x
        posList[1] = y
        imgNormWhite = copy.deepcopy(cache)
        cv2.circle(imgNormWhite, (x, y), 20, (255, 0, 0), 2)
        # cv2.destroyAllWindows()


# main
if __name__ == "__main__":

    # params
    plotta = True
    numClusters = 5

    # dirs
    dirImgs = 'imgs'
    dirTypes = 'types'
    # dirImgs = '../../uploads/original'
    dirResult = 'results'
    os.makedirs(dirResult, exist_ok=True)
    fileLabels = '../../uploads/uploadsAndEvaluations_vAng.xlsx'

    # result file
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    fileResultName = current_time + '.txt'
    fileResultNameFull = os.path.join(dirResult, fileResultName)
    fileResult = open(fileResultNameFull, "x")
    fileResult.close()

    # dir temp results
    dirNorm = os.path.join(dirImgs, 'norm')
    dirClust = os.path.join(dirImgs, 'cluster')
    dirHSV = os.path.join(dirImgs, 'hsv')
    dirMasks = os.path.join(dirImgs, 'mask')
    dirSegm = os.path.join(dirImgs, 'segm')
    dirFrames = os.path.join(dirImgs, 'frames')
    os.makedirs(dirNorm, exist_ok=True)
    os.makedirs(dirClust, exist_ok=True)
    os.makedirs(dirHSV, exist_ok=True)
    os.makedirs(dirMasks, exist_ok=True)
    os.makedirs(dirSegm, exist_ok=True)
    os.makedirs(dirFrames, exist_ok=True)

    # read labels
    # labelsPed = pd.read_excel(fileLabels)
    wb_obj = openpyxl.load_workbook(fileLabels)
    labelsPed = wb_obj.active

    # init measures
    countImg = 0
    centroids_all = []
    color_all = []
    label_all = []
    label_subTypes_all = []
    times_cluster = []
    times_featExtr = []
    times_class = []
    # loop on images
    print('Processing images...')
    filenameImg_all = [f for f in os.listdir(dirImgs) if f.endswith('jpeg')]
    for numImg, filenameImg in enumerate(filenameImg_all):

        # skip directories
        if os.path.isdir(os.path.join(dirImgs, filenameImg)) or not(filenameImg.endswith(".jpeg")):
            continue

        # update count
        countImg = countImg + 1

        # -----------------
        # if countImg > 4:
            # break
        # -----------------

        # seed
        cv2.setRNGSeed(42)

        # if numImg != 3:
            # continue

        # display
        util.print_pers('Image {0}: {1}'.format(countImg, filenameImg), fileResultNameFull)

        # read image
        img = cv2.imread(os.path.join(dirImgs, filenameImg))

        # resize
        # img = util.resize(img, scale_percent=50)

        # k-means clustering
        # print('\tFirst clustering')
        # labels, centers = functions.kmeans(img, 10, False,
                                      # os.path.join(dirNorm, filenameImg))

        start = time.time()

        # white normalization
        print('\tNormalization')
        imgNormWhite = functions.whiteNorm(img, plotta,
                                           os.path.join(dirNorm, filenameImg))

        # 2nd k-means clustering
        print('\tSecond clustering')
        labels, centers = functions.kmeans(imgNormWhite, numClusters, plotta,
                                           os.path.join(dirClust, filenameImg))

        end = time.time()
        times_cluster.append(end-start)

        # select input area
        print('\tChoose clustering')
        #Set mouse CallBack event
        titleStr = 'Click on the area to analyze (press ESC to stop)'
        cv2.namedWindow(titleStr)
        cv2.moveWindow(titleStr, 100, 100)
        cv2.setWindowProperty(titleStr, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(titleStr, getxy)
        # cv2.setMouseCallback(titleStr, draw_rect)
        cache = copy.deepcopy(imgNormWhite)
        while True:
            cv2.imshow(titleStr, imgNormWhite)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
        # print()

        start = time.time()

        # select cluster based on input
        mask_chosen, closestCenter, closestCluster = \
            functions.chooseClusterInteractive(imgNormWhite, posList, centers, labels)

        # morph proc
        print('\tMorph processing')
        mask_final = functions.morphProcMask(mask_chosen, plotta,
                                             os.path.join(dirMasks, filenameImg))

        # segm
        print('\tSegmentation')
        mask3 = np.dstack([mask_final]*3)  # Make it 3 channel
        img_mask = cv2.bitwise_and(imgNormWhite, mask3)
        cv2.imwrite(os.path.join(dirSegm, filenameImg), img_mask)

        # predict based on color mean
        # print('\tPrediction')
        # output = functions.predictColorSimilarity(dirTypes, imgNormWhite, mask_final)

        # check label
        print('\tChecking label')
        filenameImg_clean = os.path.splitext(filenameImg)[0]
        label, label_subTypes = functions.getLabel(labelsPed, filenameImg_clean)
        # if label != -1:  # filename is found in label file
            # count = count + 1
            # if output.lower() == label.lower():
                # runningCorrects = runningCorrects + 1

        # display
        util.print_pers('\t\tLabel: {0}'.format(label.lower()),
                        fileResultNameFull)
        util.print_pers('\t\tLabel (subtypes): {0}'.format(label_subTypes.lower()),
                        fileResultNameFull)
        # util.print_pers('\t\tOutput: {0}'.format(output.lower()),
                        # fileResultNameFull)

        # display img
        functions.displayImg(imgNormWhite, mask3, img_mask, filenameImg, label, label_subTypes, dirFrames)

        # print plot3d
        centroids, color = functions.plot3d(imgNormWhite, mask_final, label)
        centroids_all.append(centroids)
        color_all.append(color)

        # label 2 number
        label_number, label_subTypes_number = functions.label2number(label, label_subTypes)
        label_all.append(label_number)
        label_subTypes_all.append(label_subTypes_number)

        end = time.time()
        times_featExtr.append(end-start)

        # fine img
        # del labels, centers, img, mask_chosen, closestCenter, closestCluster
        print()

    # 3d scatterplot
    # functions.plot3d_all(centroids_all, color_all)

    # avg time for feat extr
    util.print_pers("Avg time cluster: {0}".format(np.mean(times_cluster)), fileResultNameFull)
    util.print_pers("Avg time feature extraction: {0}".format(np.mean(times_featExtr)), fileResultNameFull)

    # lin reg
    filenameImgType_all = os.listdir(dirTypes)
    runningCorrects = 0.0
    runningCorrects_subTypes = 0.0
    x_test = np.zeros((1, 6))
    y_test = np.zeros((1), dtype=int)
    y_subTypes_test = np.zeros((1), dtype=int)

    outputs_all = np.zeros((len(centroids_all)), dtype=int)
    prob_all = np.zeros((len(centroids_all)), dtype=float)

    outputs_subTypes_all = np.zeros((len(centroids_all)), dtype=int)
    prob_subTypes_all = np.zeros((len(centroids_all)), dtype=float)

    for num_centroid_test, centroid_single_test in enumerate(centroids_all):

        start = time.time()

        # test
        x_test[0, :] = centroid_single_test['yuv'] + centroid_single_test['rgb']
        y_test[0] = label_all[num_centroid_test]
        y_subTypes_test[0] = label_subTypes_all[num_centroid_test]

        # train
        # x_train = np.zeros((len(centroids_all)-1+len(filenameImgType_all), 6))
        # y_train = np.zeros((len(centroids_all)-1+len(filenameImgType_all)), dtype=int)
        x_train = np.zeros((len(filenameImgType_all), 6))
        y_train = np.zeros((len(filenameImgType_all)), dtype=int)
        y_subTypes_train = np.zeros((len(filenameImgType_all)), dtype=int)
        count_train = 0

        """
        for num_centroid_train, centroid_single_train in enumerate(centroids_all):
            if num_centroid_train == num_centroid_test:
                continue
            else:
                count_train = count_train + 1
            x_train[count_train, :] = centroid_single_train['yuv'] + centroid_single_train['rgb']
            color_single = color_all[num_centroid_train]
            if color_single == 'r':
                label_train = 0
            else:
                label_train = 1
            y_train[count_train] = label_train

        count_train = count_train + 1
        """

        # extract features and labels
        for numImgType, filenameImgType in enumerate(filenameImgType_all):

            imgType = cv2.imread(os.path.join(dirTypes, filenameImgType))
            labelType = filenameImgType.split('_')[0]
            labelType_subTypes = filenameImgType.split('.')[0]
            label_number, label_subTypes_number = functions.label2number(labelType, labelType_subTypes)

            h, s, i, r, g, b, y, u, v = functions.getChannelsColor(imgType)

            x_train[count_train, :] = [cv2.mean(y)[0], cv2.mean(u)[0], cv2.mean(v)[0],
                                      cv2.mean(r)[0], cv2.mean(g)[0], cv2.mean(b)[0]]
            y_train[count_train] = label_number
            y_subTypes_train[count_train] = label_subTypes_number
            count_train = count_train + 1

        # get max distance between prototypes
        # distC = distance_matrix(x_train, x_train)
        # max_dist = distC.max()
        max_dist = 300

        # train linear class
        # logreg = LogisticRegression()
        # logreg.fit(x_train, y_train)
        clf = neighbors.KNeighborsClassifier(1, weights="uniform")
        clf_subTypes = neighbors.KNeighborsClassifier(1, weights="uniform")
        clf.fit(x_train, y_train)
        clf_subTypes.fit(x_train, y_subTypes_train)

        # output_num = logreg.predict(x_test)[0]
        output_num = clf.predict(x_test)[0]
        output_subTypes_num = clf_subTypes.predict(x_test)[0]

        outputs_all[num_centroid_test] = output_num
        outputs_subTypes_all[num_centroid_test] = output_subTypes_num

        output_dist = clf.kneighbors(x_test, 1, return_distance=True)
        output_subTypes_dist = clf_subTypes.kneighbors(x_test, 1, return_distance=True)

        prob_all[num_centroid_test] = 1 - float(output_dist[0]) / max_dist
        prob_subTypes_all[num_centroid_test] = 1 - float(output_subTypes_dist[0]) / max_dist

        # check
        if output_num == y_test[0]:
            runningCorrects = runningCorrects + 1
        if output_subTypes_num == y_subTypes_test[0]:
            runningCorrects_subTypes = runningCorrects_subTypes + 1

        # update img
        filenameImg = filenameImg_all[num_centroid_test]
        # img = Image.open(os.path.join(dirFrames, filenameImg))
        img = cv2.imread(os.path.join(dirFrames, filenameImg))
        output, output_subTypes = functions.number2label(output_num, output_subTypes_num)
        functions.updateImg(img, output, output_subTypes, filenameImg, dirFrames)


        end = time.time()
        times_class.append(end-start)

    # avg time for feat extr
    util.print_pers("Avg time classification: {0}".format(np.mean(times_class)), fileResultNameFull)

    # accuracy
    accuracy = runningCorrects / len(centroids_all)
    accuracy_subTypes = runningCorrects_subTypes / len(centroids_all)
    util.print_pers('', fileResultNameFull)
    util.print_pers('Accuracy (%): {0}'.format(accuracy*100), fileResultNameFull)
    util.print_pers('Accuracy (subtypes) (%): {0}'.format(accuracy_subTypes*100), fileResultNameFull)

    #
    listIm = []
    for numImg, filenameImg in enumerate(filenameImg_all):
        if os.path.isfile(os.path.join(dirFrames, filenameImg)):
            if numImg == 0:
                imgOrig = Image.open(os.path.join(dirFrames, filenameImg))
            else:
                img = Image.open(os.path.join(dirFrames, filenameImg))
                listIm.append(img)
    pdf_filename = os.path.join(dirResult, current_time + '.pdf')
    imgOrig.save(pdf_filename, "PDF", resolution=100.0, save_all=True, append_images=listIm)


    #
    print()

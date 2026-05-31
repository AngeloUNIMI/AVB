import numpy as np
from skimage.measure import label


def getLargestCC(segmentation):

    labels = label(segmentation)
    assert(labels.max() != 0) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    return largestCC

import numpy as np
import cv2


np.random.seed(42)


def kmeans(img, K, plotta, filename):

    # reshape
    Z = np.float32(img.reshape((-1, 3)))

    # K = 10
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape((img.shape[:-1]))
    reduced = np.uint8(centers)[labels]

    if plotta:
        result = [np.hstack([img, reduced])]
        for i, c in enumerate(centers):
            mask = cv2.inRange(labels, i, i)
            mask = np.dstack([mask]*3)  # Make it 3 channel
            ex_img = cv2.bitwise_and(img, mask)
            ex_reduced = cv2.bitwise_and(reduced, mask)
            result.append(np.hstack([ex_img, ex_reduced]))

        cv2.imwrite(filename, np.vstack(result))

    return labels, centers
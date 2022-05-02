import cv2


def chooseCluster(centers, labels, ref):

    highestHue = -1
    for i, c in enumerate(centers):
        mask = cv2.inRange(labels, i, i)
        h_mask = cv2.bitwise_and(ref, mask)

        # h_mask = cv2.bitwise_and(h_mask, mask_ref)

        h_mask_mean = h_mask.mean()

        print(h_mask_mean)

        if h_mask_mean > highestHue:
            highestHue = h_mask_mean
            closestCluster = i
            closestCenter = c
            mask_chosen = mask

    return mask_chosen, closestCenter, closestCluster

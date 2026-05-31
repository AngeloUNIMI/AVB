import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def plot3d_all(centroids_allImg, colors_allImg):

    fig = plt.figure()

    for numType, type in enumerate(['rgb', 'hsi', 'yuv']):

        ax = fig.add_subplot(1, 3, numType+1, projection='3d')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(type)

        for numImg, centroids in enumerate(centroids_allImg):

            color = colors_allImg[numImg]

            centroid_chosen = centroids[type]

            ax.scatter(centroid_chosen[0], centroid_chosen[1], centroid_chosen[2], c=color, marker='o')

    # legend
    legend_elements = [Line2D([0], [0], marker='o', color='r', label='Abnormal',
                              markerfacecolor='r', markersize=5),
                       Line2D([0], [0], marker='o', color='b', label='Normal',
                              markerfacecolor='b', markersize=5),
                       ]
    ax.legend(handles=legend_elements, loc='best')

    # show
    plt.show()
    fig.savefig('plot3d.jpg')

    return -1

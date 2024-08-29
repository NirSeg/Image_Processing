import cv2
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=7)
# size of the image
m,n = 921, 750

# frame points of the blank wormhole image
src_points = np.float32([[0, 0],
                            [int(n / 3), 0],
                            [int(2 * n /3), 0],
                            [n, 0],
                            [n, m],
                            [int(2 * n / 3), m],
                            [int(n / 3), m],
                            [0, m]])

# blank wormhole frame points
dst_points = np.float32([[96, 282],
                       [220, 276],
                       [344, 276],
                       [468, 282],
                       [474, 710],
                       [350, 744],
                       [227, 742],
                       [103, 714]]
                      )


def find_transform(pointset1, pointset2):
    srcX = np.zeros((2*pointset1.shape[0], 8))
    dstX = np.zeros((2*pointset2.shape[0], 1))
    length = pointset1.shape[0]
    for i in range(length):
        # line 2i
        srcX[2*i, 0] = pointset1[i, 0]
        srcX[2*i, 1] = pointset1[i, 1]
        srcX[2*i, 4] = 1
        srcX[2*i, 6] = -1*pointset1[i, 0]*pointset2[i, 0]
        srcX[2*i, 7] = -1*pointset1[i, 1]*pointset2[i, 0]
        # line 2i+1
        srcX[2*i+1, 2] = pointset1[i, 0]
        srcX[2*i+1, 3] = pointset1[i, 1]
        srcX[2*i+1, 5] = 1
        srcX[2*i+1, 6] = -1*pointset1[i, 0]*pointset2[i, 1]
        srcX[2*i+1, 7] = -1*pointset1[i, 1]*pointset2[i, 1]
    for i in range(length):
        # line 2i
        dstX[2*i] = pointset2[i, 0]
        # line 2i+1
        dstX[2*i+1] = pointset2[i, 1]
    vals = np.matmul(np.linalg.pinv(srcX), dstX)
    T = np.array([vals[0, 0], vals[1, 0], vals[4, 0], vals[2, 0], vals[3, 0], vals[5, 0], vals[6, 0], vals[7, 0], 1])
    T = np.reshape(T, (3, 3))
    return T


def trasnform_image(image, T):
    length = image.shape[0]
    width = image.shape[1]
    new_image = np.zeros((length, width), dtype=np.float32)
    T_inverse = np.linalg.pinv(T)
    vpoint = np.ones((3, 1), dtype=np.float32)
    for x_tag in range(length):
        for y_tag in range(width):
            vpoint[0][0] = y_tag
            vpoint[1][0] = x_tag
            point = np.matmul(T_inverse, vpoint)
            y_org = round(point[0][0]/point[2][0])
            x_org = round(point[1][0]/point[2][0])
            if 0 <= x_org <= length - 1 and 0 <= y_org <= width - 1:
                new_image[x_tag][y_tag] = image[x_org][y_org]
    return new_image


def create_wormhole(im, T, iter=5):
    new_image = im.copy()
    for i in range(iter):
        im = trasnform_image(im, T)
        new_image = new_image + im
    new_image = np.clip(new_image, 0, 255)
    return new_image
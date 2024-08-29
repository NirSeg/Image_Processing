import cv2
import numpy as np
import scipy.signal
from scipy.signal import convolve2d as conv
from scipy.ndimage.filters import gaussian_filter as gaussian
import matplotlib.pyplot as plt


def sobel(im):
    maskx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    masky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8
    sobelx = abs(conv(im, maskx))
    sobely = abs(conv(im, masky))
    sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    ret, new_im = cv2.threshold(sobel, 7, 14, cv2.THRESH_BINARY)
    return new_im


def canny(im):
    blur = gaussian(im, sigma=2)
    return cv2.Canny(blur, 40, 180)


def hough_circles(im):
    im_c = im.copy()
    circles = cv2.HoughCircles(im_c,cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=30, minRadius=30, maxRadius=45)
    for i in circles[0, :]:
        cv2.circle(im_c, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 3)
    return im_c


def hough_lines(im):
    im_l = im.copy()
    dst = cv2.Canny(im, 200, 300)
    # return dst
    lines = cv2.HoughLines(dst, 1, np.pi/180, 180)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(im_l, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return im_l

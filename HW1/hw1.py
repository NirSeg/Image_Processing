import numpy as np
import matplotlib.pyplot as plt
import cv2

def histImage(im):
    h = np.zeros(256)
    for i in range(len(im)):
        for j in range(len(im[0])):
            h[int(im[i, j])] += 1
    return h


def nhistImage(im):
    nh = histImage(im)
    N = im.shape[0]*im.shape[1]
    nh /= N
    return nh


def ahistImage(im):
    ah = np.cumsum(histImage(im))
    return ah


def calcHistStat(h):
    arr = np.array(range(256))
    m = np.matmul(h/np.sum(h), arr)
    v = np.matmul(h/np.sum(h), np.square(arr)) - m**2
    return m, v


def mapImage(im,tm):
    nim = tm[im]
    nim[nim < 0] = 0
    nim[nim > 255] = 255
    return nim


def histEqualization(im):
    ah = ahistImage(im)
    N = im.shape[0]*im.shape[1] / 256
    tm = np.ceil(ah/N)
    return tm


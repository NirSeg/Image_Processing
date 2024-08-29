import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import convolve2d

# the copy in the first lines of the function is so that you don't ruin
# the original image. it will create a new one. 

def add_SP_noise(im, p):
    sp_noise_im = im.copy()
    length = im.shape[0]
    width = im.shape[1]
    num_pix_noise = int(length * width * p)
    pixels_noise = random.sample(range(length * width), num_pix_noise)
    sp_noise_im = np.ravel(sp_noise_im)
    sp_noise_im[pixels_noise[:num_pix_noise // 2]] = 0
    sp_noise_im[pixels_noise[num_pix_noise // 2:]] = 255
    sp_noise_im = np.reshape(sp_noise_im, (length, width))
    return sp_noise_im


def clean_SP_noise_single(im, radius):
    noise_im = im.copy()
    clean_im = im.copy()
    length = im.shape[0]
    width = im.shape[1]
    for i in range(radius, length - radius):
        for j in range(radius, width - radius):
            clean_im[i, j] = np.median(noise_im[i-radius:i+radius+1, j-radius:j+radius+1])
    return clean_im


def clean_SP_noise_multiple(images):
    clean_image = np.median(images, axis=0)
    return clean_image


def add_Gaussian_Noise(im, s):
    gaussian_noise_im = im.copy()
    length = im.shape[0]
    width = im.shape[1]
    gaussian_noise = np.random.normal(0, s, size=(length, width))
    gaussian_noise_im = gaussian_noise_im + gaussian_noise
    return gaussian_noise_im


def clean_Gaussian_noise(im, radius, maskSTD):
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    exponent = -np.divide((np.square(X) + np.square(Y)), (2 * np.square(maskSTD)))
    gaussian_mask = np.exp(exponent, dtype=np.float)
    gaussian_mask = np.divide(gaussian_mask, np.sum(gaussian_mask))
    cleaned_im = scipy.signal.convolve2d(im, gaussian_mask, mode='same')
    return cleaned_im.astype(np.uint8)


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()
    length = im.shape[0]
    width = im.shape[1]
    for i in range(radius, length - radius):
        for j in range(radius, width - radius):
            X, Y = np.meshgrid(np.arange(-radius + i, radius + i + 1), np.arange(-radius + j, radius + j + 1))
            window = im[i - radius:i + radius + 1, j - radius:j + radius + 1]

            giExponent = -np.divide(np.square(window - im[i, j]), (2 * np.square(stdIntensity)), dtype=np.float)
            gi = np.exp(giExponent, dtype=np.float)
            gi = np.divide(gi, np.sum(gi))

            gsExponent = -np.divide((np.square(X - i) + np.square(Y - j)), (2 * np.square(stdSpatial)), dtype=np.float)
            gs = np.exp(gsExponent, dtype=np.float)
            gs = np.divide(gs, np.sum(gs))

            bilateral_im[i, j] = np.divide(np.sum(gi * gs * window), np.sum(gi * gs))
    return bilateral_im.astype(np.uint8)



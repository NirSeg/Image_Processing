import numpy as np
import scipy.signal
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2

def contrast(im, range):
    new = range[1] - range[0]
    old = np.max(im) - np.min(im)
    a = new / old
    b = range[1] - a * np.max(im)
    nim = a * im + b
    return nim

# for baby
src_baby1 = np.float32([[6, 20], [111, 20], [111, 130], [5, 130]])
src_baby2 = np.float32([[181, 5], [249, 70], [176.5, 120], [121, 51]])
src_baby3 = np.float32([[78, 162.5], [146.5, 117], [245, 160], [131.5, 244]])
dst_baby = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])


def find_transform(pointset1, pointset2):
    srcX = np.zeros((2 * pointset1.shape[0], 8))
    dstX = np.zeros((2 * pointset2.shape[0], 1))
    length = pointset1.shape[0]
    for i in range(length):
        # line 2i
        srcX[2 * i, 0] = pointset1[i, 0]
        srcX[2 * i, 1] = pointset1[i, 1]
        srcX[2 * i, 4] = 1
        srcX[2 * i, 6] = -1 * pointset1[i, 0] * pointset2[i, 0]
        srcX[2 * i, 7] = -1 * pointset1[i, 1] * pointset2[i, 0]
        # line 2i+1
        srcX[2 * i + 1, 2] = pointset1[i, 0]
        srcX[2 * i + 1, 3] = pointset1[i, 1]
        srcX[2 * i + 1, 5] = 1
        srcX[2 * i + 1, 6] = -1 * pointset1[i, 0] * pointset2[i, 1]
        srcX[2 * i + 1, 7] = -1 * pointset1[i, 1] * pointset2[i, 1]
    for i in range(length):
        # line 2i
        dstX[2 * i] = pointset2[i, 0]
        # line 2i+1
        dstX[2 * i + 1] = pointset2[i, 1]
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
            y_org = round(point[0][0] / point[2][0])
            x_org = round(point[1][0] / point[2][0])
            if 0 <= x_org <= length - 1 and 0 <= y_org <= width - 1:
                new_image[x_tag][y_tag] = image[x_org][y_org]
    return new_image


def clean_SP_noise_single(im, radius):
    noise_im = im.copy()
    clean_im = im.copy()
    length = im.shape[0]
    width = im.shape[1]
    for i in range(radius, length - radius):
        for j in range(radius, width - radius):
            clean_im[i, j] = np.median(noise_im[i - radius:i + radius + 1, j - radius:j + radius + 1])
    return clean_im


def clean_SP_noise_multiple(images):
    clean_image = np.median(images, axis=0)
    return clean_image

# for watermelon
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])


# the functions

def clean_baby(im):
    T1 = find_transform(src_baby1, dst_baby)
    T2 = find_transform(src_baby2, dst_baby)
    T3 = find_transform(src_baby3, dst_baby)
    im1 = trasnform_image(im, T1)
    im2 = trasnform_image(im, T2)
    im3 = trasnform_image(im, T3)
    clean_im1 = cv2.medianBlur(clean_SP_noise_single(im1, radius=1), 5)
    clean_im2 = cv2.medianBlur(clean_SP_noise_single(im2, radius=1), 5)
    clean_im3 = cv2.medianBlur(clean_SP_noise_single(im3, radius=1), 5)
    clean_im = clean_SP_noise_multiple([clean_im1, clean_im2, clean_im3])
    return clean_im


def clean_windmill(im):
    freq_im = np.fft.fftshift(np.fft.fft2(im))
    freq_im[124, 100] = 0
    freq_im[132, 156] = 0
    clean_im = abs(np.fft.ifft2(np.fft.ifftshift(freq_im)))
    return clean_im


def clean_watermelon(im):
    clean_im = cv2.filter2D(im, -1, kernel_sharpening)
    return clean_im


def clean_umbrella(im):
    num_pics = 2
    kernel = np.zeros(im.shape)
    kernel[0][0] = 1
    kernel[4][79] = 1
    freq_im = np.fft.fft2(im)
    freq_kernel = np.fft.fft2(kernel)
    freq_kernel[np.abs(freq_kernel) < 0.01] = 1
    freq_clean_im = freq_im / freq_kernel
    clean_im = num_pics * abs(np.fft.ifft2(freq_clean_im))
    return clean_im


def clean_USAflag(im):
    radius = 5
    clean_im = im.copy()
    for i in range(im.shape[0]):
        for j in range(radius, im.shape[1] - radius):
            clean_im[i][j] = np.median(im[i, j - radius: j + radius + 1])
    stars = im[0:90, 0:142]
    clean_im[0:90, 0:142] = stars
    return clean_im


def clean_cups(im):
    freq_im = np.fft.fftshift(np.fft.fft2(im))
    freq_im[108:149, 108:149] *= 2
    clean_im = abs(np.fft.ifft2(np.fft.ifftshift(freq_im)))
    return clean_im


def clean_house(im):
    num_pics = 10
    kernel = np.zeros(im.shape)
    kernel[0, :num_pics] = 1
    freq_kernel = np.fft.fft2(kernel)
    freq_im = np.fft.fft2(im)
    freq_kernel[np.abs(freq_kernel) < 0.01] = 1
    freq_clean_im = freq_im / freq_kernel
    clean_im = num_pics * abs(np.fft.ifft2(freq_clean_im))
    return clean_im


def clean_bears(im):
    return contrast(im, (0, 255))


'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images\windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_fourier = np.fft.fft2(img) # fft - remember this is a complex numbers matrix 
    img_fourier = np.fft.fftshift(img_fourier) # shift so that the DC is in the middle

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')

    plt.subplot(1,3,2)
    plt.imshow(np.log(abs(img_fourier)), cmap='gray') # need to use abs because it is complex, the log is just so that we can see the difference in values with out eyes.
    plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''

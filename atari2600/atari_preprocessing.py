import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    img = np.mean(img, axis=2).astype(np.uint8)
    img = img[34:34+160, 0:160]
    img = img[::2, ::2]
    # img = (img*255).astype(np.uint8)
    # print(np.max(img))
    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.show()
    return img
    # return to_grayscale(downsample(img))

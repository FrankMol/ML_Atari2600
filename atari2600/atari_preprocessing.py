import numpy as np
# from skimage.color import rgb2gray
# from skimage.transform import resize
# import matplotlib.pyplot as plt

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)


def downsample(img):
    return img[::2, ::2]

# def use8484(img):
#     img = rgb2gray(img)[34:34+160, 0:160]
#     img = np.uint8(resize(img, (84, 84), order=0) * 255)
#     return img

def preprocess(img):

    return to_grayscale(downsample(img))
    # return use8484(img)

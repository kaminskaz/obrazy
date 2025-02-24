import numpy as np
import cv2

def convert_to_gray(image):
    """
    Conversion to greyscale
    """
    gray = np.dot(image[..., :3], [0.114, 0.587, 0.299])
    return gray.astype(np.uint8)

def adjust_brightness(image, value):
    """
    Adjusting brightness
    """
    bright = np.clip(image.astype(np.int16) + value, 0, 255)
    return bright.astype(np.uint8)

def adjust_contrast(image, factor):
    """
    Contrast adjustment
    """
    contrasted = np.clip((image.astype(np.int16) - 128) * factor + 128, 0, 255)
    return contrasted.astype(np.uint8)

def negative(image):
    """
    Negative of an image
    """
    return 255 - image

def binarization(image, threshold=128):
    """
    Binarization of an image
    """
    gray = convert_to_gray(image)
    binary = np.where(gray > threshold, 255, 0)
    return binary.astype(np.uint8)

def adjust_saturation(image, scale):
    """
    Adjusting saturation
    """
    # TODO: can we use cv2.cvtColor here? if not then implement whole conversion algorithm

def average_filter(image, kernel_size=3):
    """
    Average filter 
    """
    pad = kernel_size // 2
    image_float = image.astype(np.float32)
    output = np.zeros_like(image_float)
    for c in range(image.shape[2]):
        padded = np.pad(image_float[..., c], pad, mode='edge')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j, c] = np.mean(region)
    return np.clip(output, 0, 255).astype(np.uint8)

def gaussian_filter(image, kernel_size=3, sigma=1.0):
    """
    Gaussian filter
    """
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    x_dist, y_dist = np.meshgrid(ax, ax)
    kernel = np.exp(-(x_dist**2 + y_dist**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    pad = kernel_size // 2
    image_float = image.astype(np.float32)
    output = np.zeros_like(image_float)
    for c in range(image.shape[2]):
        padded = np.pad(image_float[..., c], pad, mode='edge')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j, c] = np.sum(region * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)

def sharpen(image):
    """
    Sharpening filter
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2
    image_float = image.astype(np.float32)
    output = np.zeros_like(image_float)
    for c in range(image.shape[2]):
        padded = np.pad(image_float[..., c], pad, mode='edge')
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j, c] = np.sum(region * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)

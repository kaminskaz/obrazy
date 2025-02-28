import numpy as np


class ImageProcessor:
    def __init__(self, image):
        self.image = image

    def convert_to_gray(self):
        """
        Convert the image to grayscale.
        """
        gray = np.dot(self.image[..., :3], [0.114, 0.587, 0.299])
        gray = gray.astype(np.uint8)

        # Convert (H, W) → (H, W, 3) for GUI compatibility
        gray = np.stack([gray] * 3, axis=-1)  # Duplicate grayscale values across RGB channels

        return gray

    def adjust_brightness(self, value):
        """
        Adjusting brightness
        """
        bright = np.clip(self.image.astype(np.int16) + value, 0, 255)
        return bright.astype(np.uint8)

    def adjust_contrast(self, factor):
        """
        Contrast adjustment
        """
        contrasted = np.clip((self.image.astype(np.int16) - 128) * float(factor) + 128, 0, 255)
        return contrasted.astype(np.uint8)

    def negative(self):
        """
        Negative of an image
        """
        return 255 - self.image

    def binarization(self, threshold=128):
        """
        Binarization of an image
        """
        gray = self.convert_to_gray()
        binary = np.where(gray > threshold, 255, 0)
        return binary.astype(np.uint8)

    def adjust_saturation(self, scale):
        """
        Adjusting saturation
        """
        # TODO: can we use cv2.cvtColor here? if not then implement whole conversion algorithm
        pass

    def average_filter(self, kernel_size=3):
        """
        Average filter 
        """
        pad = kernel_size // 2
        image_float = self.image.astype(np.float32)
        output = np.zeros_like(image_float)
        for c in range(self.image.shape[2]):
            padded = np.pad(image_float[..., c], pad, mode='edge')
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    output[i, j, c] = np.mean(region)
        return np.clip(output, 0, 255).astype(np.uint8)

    def gaussian_filter(self, kernel_size=3, sigma=1.0):
        """
        Gaussian filter
        """
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        x_dist, y_dist = np.meshgrid(ax, ax)
        kernel = np.exp(-(x_dist**2 + y_dist**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)
        
        pad = kernel_size // 2
        image_float = self.image.astype(np.float32)
        output = np.zeros_like(image_float)
        for c in range(self.image.shape[2]):
            padded = np.pad(image_float[..., c], pad, mode='edge')
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    output[i, j, c] = np.sum(region * kernel)
        return np.clip(output, 0, 255).astype(np.uint8)

    def sharpen(self):
        """
        Sharpening filter
        """
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        image_float = self.image.astype(np.float32)
        output = np.zeros_like(image_float)
        for c in range(self.image.shape[2]):
            padded = np.pad(image_float[..., c], pad, mode='edge')
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    output[i, j, c] = np.sum(region * kernel)
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def edge_detection(self):
        """
        Edge detection
        """
        kernel = np.array([[1, 0, -1],
                           [0, 0, 0],
                           [-1, 0, 1]])
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2
        image_float = self.image.astype(np.float32)
        output = np.zeros_like(image_float)
        for c in range(self.image.shape[2]):
            padded = np.pad(image_float[..., c], pad, mode='edge')
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    output[i, j, c] = np.sum(region * kernel)
        return np.clip(output, 0, 255).astype(np.uint8)
    
    def apply_custom_filter(self, kernel):
        """Apply a custom filter to the image using the provided kernel."""
        pad = kernel.shape[0] // 2
        image_float = self.image.astype(np.float32)
        output = np.zeros_like(image_float)

        for c in range(image_float.shape[2]):
            padded = np.pad(image_float[..., c], pad, mode='edge')
            for i in range(image_float.shape[0]):
                for j in range(image_float.shape[1]):
                    region = padded[i:i + kernel.shape[0], j:j + kernel.shape[1]]
                    output[i, j, c] = np.sum(region * kernel)

        return np.clip(output, 0, 255).astype(np.uint8)


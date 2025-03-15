import numpy as np
import matplotlib.pyplot as plt


class ImageProcessor:
    def __init__(self, image):
        self.image = image
        self.kernels = {
        'average': np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]]) / 9,
        'sharpen': np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]]),
        'laplacian': np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]]),
        'edge_sobel_horizontal': np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]),
        'edge_sobel_vertical': np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]]),
        'edge_prewitt_horizontal': np.array([[1, 0, -1],
                                  [1, 0, -1],
                                  [1, 0, -1]]),
        'edge_prewitt_vertical': np.array([[1, 1, 1],
                                  [0, 0, 0],
                                  [-1, -1, -1]]),
        'gaussian_blur': np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]]) / 16,
        'roberts_cross_horizontal': np.array([[1, 0],
                                  [0, -1]]),
        'roberts_cross_vertical': np.array([[0, 1],
                                  [-1, 0]]),
        
    }

    def convert_to_gray(self):
        """
        Convert the image to grayscale.
        """
        gray = np.dot(self.image[..., :3], [0.114, 0.587, 0.299])
        gray = gray.astype(np.uint8)
        # Convert (H, W) → (H, W, 3) for GUI compatibility
        gray = np.stack([gray] * 3, axis=-1)
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
        f = (259 * (factor + 255)) / (255 * (259 - factor))
        contrasted = np.clip((self.image.astype(np.int16) - 128) * f + 128, 0, 255)
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

    def convolve(self, kernel):
        """
        Custom convolve function to apply a kernel to the image.
        """
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2

        image_padded = np.pad(self.image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        output = np.zeros_like(self.image, dtype=np.float32)

        for c in range(self.image.shape[2]):  
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    region = image_padded[i:i + kernel_size, j:j + kernel_size, c] 
                    output[i, j, c] = np.sum(region * kernel)  

        return np.clip(output, 0, 255).astype(np.uint8)

    def average_filter(self):
        """
        Apply an average filter to the image.
        """
        kernel = self.kernels['average']
        return self.convolve(kernel)
    
    def gaussian_filter(self):
        """
        Apply a Gaussian filter to the image.
        """
        kernel = self.kernels['gaussian_blur']
        return self.convolve(kernel)
    
    def sharpen(self):
        """
        Apply a sharpening filter to the image.
        """
        kernel = self.kernels['sharpen']
        return self.convolve(kernel)
    
    def edge_detection(self, method):
        """
        Apply an edge detection filter to the image according to the specified method.
        """
        if method == 'sobel':
            kernel_h = self.kernels['edge_sobel_horizontal']
            kernel_v = self.kernels['edge_sobel_vertical']
        elif method == 'prewitt':
            kernel_h = self.kernels['edge_prewitt_horizontal']
            kernel_v = self.kernels['edge_prewitt_vertical']
        elif method == 'laplacian':
            kernel = self.kernels['laplacian']
        elif method == 'roberts_cross':
            kernel_h = self.kernels['roberts_cross_horizontal']
            kernel_v = self.kernels['roberts_cross_vertical']
        
        if method == 'sobel' or method == 'roberts_cross' or method == 'prewitt':
            output_h = self.convolve(kernel_h)
            output_v = self.convolve(kernel_v)
            # output = np.sqrt(output_h ** 2 + output_v ** 2) for some reason not working 
            output = abs(output_h) + abs(output_v)
            
        else:
            output = self.convolve(kernel)
        return output
        
    def opening(self, kernel=None):
        """
        Opening of an image
        """
        if kernel is None:
            kernel = self.get_kernel_bin("square")  

        eroded_img = self.convolute_binary(self.image, kernel_shape="square", operation="erosion")

        return self.convolute_binary(eroded_img, kernel_shape="square", operation="dilation") 

    def closing(self, kernel=None):  
        """
        Closing of an image
        """
        if kernel is None:
            kernel = self.get_kernel_bin("square") 

        dilated_img = self.convolute_binary(self.image, kernel_shape="square", operation="dilation")

        return self.convolute_binary(dilated_img, kernel_shape="square", operation="erosion")
    
    def get_kernel_bin(self, shape="square"):
        """Generate a kernel based on the selected shape."""
        if shape == "square":
            return np.zeros((3, 3), dtype=np.uint8)
        elif shape == "cross":
            kernel = np.ones((3, 3), dtype=np.uint8)
            kernel[1, :] = 0
            kernel[:, 1] = 0
            return kernel
        elif shape == "vertical_line":
            return np.array([[0, 1, 0],
                             [0, 1, 0],
                             [0, 1, 0]])
        elif shape == "horizontal_line":
            return np.array([[0, 0, 0],
                             [1, 1, 1],
                             [0, 0, 0]])
        else:
            raise ValueError(f"Unknown kernel shape: {shape}")

    def convolute_binary(self, image=None, kernel_shape="square", operation="dilation"):
        """
        Perform binary image convolution for either dilation or erosion depending on the operation parameter.
        
        Parameters:
            image (np.array): The binary image to process (default is the class image).
            kernel_shape (str): The shape of the kernel to use (e.g., "square", "cross", etc.).
            operation (str): The operation to perform: 'dilation' for dilation, 'erosion' for erosion.
            
        Returns:
            np.array: The processed image after applying the selected operation.
        """
        kernel = self.get_kernel_bin(kernel_shape)
        if image is None:
            image = self.image

        kernel_size = kernel.shape[0]
        pad = kernel_size // 2

        image_padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        if operation == "dilation":
            output = np.ones_like(image, dtype=np.uint8) * 255 
        elif operation == "erosion":
            output = np.ones_like(image, dtype=np.uint8) * 255
        else:
            raise ValueError(f"Invalid operation: {operation}. Use 'dilation' or 'erosion'.")
        output = np.pad(output, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = image_padded[i:i + kernel_size, j:j + kernel_size]

                if operation == "dilation":
                    if region[1, 1, 1] == 0:
                        output[i, j] = 0
                        for k in range(kernel_size):
                            for l in range(kernel_size):
                                if kernel[k, l] == 0:
                                    output[i + k, j + l] = 0

                elif operation == "erosion":
                    erosion_condition = True
                    for k in range(kernel_size):
                        for l in range(kernel_size):
                            if kernel[k, l] == 0 and region[1, k, l].any() != 0:  
                                erosion_condition = False
                                break
                        if not erosion_condition:
                            break

                    if not erosion_condition:
                        output[i, j] = 255
                    else:
                        output[i, j] = 0

        return output[pad:-pad, pad:-pad]

    def apply_custom_filter(self, kernel):
        """
        Apply a custom filter to the image using the provided kernel.
        """
        return self.convolve(kernel)

    def rotate(self, angle):
        """
        Rotate the image by the given angle 
        """
        theta = np.radians(angle)
        
        h, w, c = self.image.shape  

        center_x, center_y = w // 2, h // 2

        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        corners = np.array([
            [-center_x, -center_y],
            [w - center_x, -center_y],
            [-center_x, h - center_y],
            [w - center_x, h - center_y]
        ])

        new_corners = np.dot(corners, rotation_matrix.T)
        min_x, min_y = new_corners.min(axis=0)
        max_x, max_y = new_corners.max(axis=0)

        new_w = int(np.ceil(max_x - min_x))
        new_h = int(np.ceil(max_y - min_y))

        rotated_image = np.zeros((new_h, new_w, c), dtype=self.image.dtype)

        new_center_x, new_center_y = new_w // 2, new_h // 2

        for i in range(new_h):
            for j in range(new_w):
                x, y = np.dot(rotation_matrix.T, np.array([j - new_center_x, i - new_center_y]))
                x, y = int(round(x + center_x)), int(round(y + center_y))
                if 0 <= x < w and 0 <= y < h:
                    rotated_image[i, j] = self.image[y, x]

        return rotated_image
    
    def flip(self, direction):
        """
        Flip the image in the specified direction. 
        direction: 'horizontal' or 'vertical
        '"""
        height, width, channels = self.image.shape
        
        if direction == 'horizontal':
            flipped_image = self.image.copy()
            for row in range(height):
                flipped_image[row] = self.image[row, ::-1] 

        elif direction == 'vertical':
            flipped_image = self.image.copy()
            flipped_image = flipped_image[::-1, :, :]
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")
        return flipped_image
        


    




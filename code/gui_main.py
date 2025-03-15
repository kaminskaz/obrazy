from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import sys
import cv2
import numpy as np
from image_processor import ImageProcessor 
from gui_welcome import WelcomeDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig) 
        self.axes = fig.add_subplot(111)
        self.axes.set_title("")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        # self.axes.set_frame_on(False) 
        self.axes.spines['top'].set_visible(False)  
        self.axes.spines['right'].set_visible(False)  
        self.axes.spines['left'].set_visible(False) 
        self.axes.spines['bottom'].set_visible(False)
        fig.tight_layout() 
        self.draw()
    
class ImageProcessorGUI(QWidget):  
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bildverarbeitung - Image Processing")
        self.image_path = None  
        self.image_processor = None  
        self.processed_image = None  
        self.binary_kernel = None

        self.big_layout = QHBoxLayout()
        # Menu bar
        self.create_menu()

        # Main layout
        self.main_layout = QVBoxLayout()

        # Image layout
        self.image_layout = QHBoxLayout()

        # Original image
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setFixedSize(400, 400)
        self.original_label.setStyleSheet("border: 2px solid black;")

        # Processed image
        self.processed_label = QLabel("Modified Image")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setFixedSize(400, 400)
        self.processed_label.setStyleSheet("border: 2px solid black;")

        # Add labels to layout
        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.processed_label)

        # Sliders layout
        self.slider_layout = QVBoxLayout()
        
        # Brightness slider
        self.brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(-255)
        self.brightness_slider.setMaximum(255)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.apply_brightness)
        self.brightness_slider.valueChanged.connect(self.update_histogram)
        self.brightness_slider.valueChanged.connect(self.update_projection)

        # Contrast slider
        self.contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setMinimum(-255)
        self.contrast_slider.setMaximum(255)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.apply_contrast)
        self.contrast_slider.valueChanged.connect(self.update_histogram)
        self.contrast_slider.valueChanged.connect(self.update_projection)

        # Binarization slider
        self.binarization_label = QLabel("Binarization Threshold:")
        self.binarization_slider = QSlider(Qt.Orientation.Horizontal)
        self.binarization_slider.setMinimum(0)
        self.binarization_slider.setMaximum(255)
        self.binarization_slider.setValue(128) 
        self.binarization_slider.valueChanged.connect(self.binarization)
        self.binarization_slider.valueChanged.connect(self.update_histogram)
        self.binarization_slider.valueChanged.connect(self.update_projection)

        # Add sliders to the layout
        self.slider_layout.addWidget(self.brightness_label)
        self.slider_layout.addWidget(self.brightness_slider)
        self.slider_layout.addWidget(self.contrast_label)
        self.slider_layout.addWidget(self.contrast_slider)
        self.slider_layout.addWidget(self.binarization_label)
        self.slider_layout.addWidget(self.binarization_slider)
        self.slider_layout.setSpacing(4)

        # Buttons layout
        self.button_layout = QVBoxLayout()

        # Fast filter buttons
        self.gray_button = QPushButton("Convert to Gray")
        self.gray_button.clicked.connect(self.convert_to_gray)
        self.gray_button.clicked.connect(self.update_histogram)
        self.gray_button.clicked.connect(self.update_projection)
        self.gray_button.setStyleSheet("background-color: lightpink; color: black;")  # Light pink with black text
        self.button_layout.addWidget(self.gray_button)

        self.negative_button = QPushButton("Negative")
        self.negative_button.clicked.connect(self.negative)
        self.negative_button.clicked.connect(self.update_histogram)
        self.negative_button.clicked.connect(self.update_projection)
        self.negative_button.setStyleSheet("background-color: lightpink; color: black;")  # Light pink with black text
        self.button_layout.addWidget(self.negative_button)

        # Slow filter buttons
        self.button_layout_slow = QHBoxLayout()

        self.average_filter_button = QPushButton("Average Filter")
        self.average_filter_button.clicked.connect(self.apply_average_filter)
        self.average_filter_button.clicked.connect(self.update_histogram)
        self.average_filter_button.clicked.connect(self.update_projection)
        self.average_filter_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_slow.addWidget(self.average_filter_button)

        self.gaussian_filter_button = QPushButton("Gaussian Filter")
        self.gaussian_filter_button.clicked.connect(self.apply_gaussian_filter)
        self.gaussian_filter_button.clicked.connect(self.update_histogram)
        self.gaussian_filter_button.clicked.connect(self.update_projection)
        self.gaussian_filter_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_slow.addWidget(self.gaussian_filter_button)

        self.sharpen_button = QPushButton("Sharpen")
        self.sharpen_button.clicked.connect(self.apply_sharpen)
        self.sharpen_button.clicked.connect(self.update_histogram)
        self.sharpen_button.clicked.connect(self.update_projection)
        self.sharpen_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_slow.addWidget(self.sharpen_button)

        # Edge Detection buttons
        self.button_layout_edge = QHBoxLayout()

        self.edge_detection_button_sobel = QPushButton("Sobel")
        self.edge_detection_button_sobel.clicked.connect(lambda: self.apply_edge_detection(method='sobel'))
        self.edge_detection_button_sobel.clicked.connect(self.update_histogram)
        self.edge_detection_button_sobel.clicked.connect(self.update_projection)
        self.edge_detection_button_sobel.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_edge.addWidget(self.edge_detection_button_sobel)

        self.edge_detection_prewitt_button = QPushButton("Prewitt")
        self.edge_detection_prewitt_button.clicked.connect(lambda: self.apply_edge_detection(method='prewitt'))
        self.edge_detection_prewitt_button.clicked.connect(self.update_histogram)
        self.edge_detection_prewitt_button.clicked.connect(self.update_projection)
        self.edge_detection_prewitt_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_edge.addWidget(self.edge_detection_prewitt_button)

        self.edge_detection_button_laplacian = QPushButton("Laplacian")
        self.edge_detection_button_laplacian.clicked.connect(lambda: self.apply_edge_detection(method='laplacian'))
        self.edge_detection_button_laplacian.clicked.connect(self.update_histogram)
        self.edge_detection_button_laplacian.clicked.connect(self.update_projection)
        self.edge_detection_button_laplacian.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_edge.addWidget(self.edge_detection_button_laplacian)
        
        self.edge_detection_button_roberts = QPushButton("Roberts")
        self.edge_detection_button_roberts.clicked.connect(lambda: self.apply_edge_detection(method='roberts_cross'))
        self.edge_detection_button_roberts.clicked.connect(self.update_histogram)
        self.edge_detection_button_roberts.clicked.connect(self.update_projection)
        self.edge_detection_button_roberts.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_edge.addWidget(self.edge_detection_button_roberts)

        # Morphological Operations
        self.morphological_operations_layout = QHBoxLayout()
        #add dropdown to choose kernel shape
        self.kernel_shape_label = QLabel("Select Kernel Shape:")
        self.kernel_shape_combo = QComboBox()
        self.kernel_shape_combo.addItems(["square", "cross", "vertical_line", "horizontal_line"])
        self.morphological_operations_layout.addWidget(self.kernel_shape_label)
        self.morphological_operations_layout.addWidget(self.kernel_shape_combo)
        #when chosen shape, update self.binary_kernel
        self.kernel_shape_combo.currentIndexChanged.connect(self.update_kernel_shape)

        self.erosion_button = QPushButton("Errosion")
        self.erosion_button.clicked.connect(self.erosion)
        self.erosion_button.clicked.connect(self.update_histogram)
        self.erosion_button.clicked.connect(self.update_projection)
        self.erosion_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.morphological_operations_layout.addWidget(self.erosion_button)

        self.dilation_button = QPushButton("Dilation")
        self.dilation_button.clicked.connect(self.dilation)
        self.dilation_button.clicked.connect(self.update_histogram)
        self.dilation_button.clicked.connect(self.update_projection)
        self.dilation_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.morphological_operations_layout.addWidget(self.dilation_button)

        self.opening_button = QPushButton("Opening")
        self.opening_button.clicked.connect(self.opening)
        self.opening_button.clicked.connect(self.update_histogram)
        self.opening_button.clicked.connect(self.update_projection)
        self.opening_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.morphological_operations_layout.addWidget(self.opening_button)

        self.closing_button = QPushButton("Closing")
        self.closing_button.clicked.connect(self.closing)
        self.closing_button.clicked.connect(self.update_histogram)
        self.closing_button.clicked.connect(self.update_projection)
        self.closing_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.morphological_operations_layout.addWidget(self.closing_button)

        # Custom Filter Layout
        self.custom_filter_layout = QVBoxLayout()
        self.custom_label = QLabel("Custom kernel filter")
        self.custom_label.setStyleSheet("font-weight: bold;")
        self.custom_filter_layout.addWidget(self.custom_label)

        # Kernel Size Selection
        self.kernel_size_label = QLabel("Select Kernel Size:")
        self.kernel_size_combo = QComboBox()
        self.kernel_size_combo.addItems(["3", "5", "7"])
        self.kernel_size_combo.currentIndexChanged.connect(self.update_kernel_input_grid)
        self.custom_filter_layout.addWidget(self.kernel_size_label)
        self.custom_filter_layout.addWidget(self.kernel_size_combo)

        # Kernel Input Grid
        self.kernel_input_grid = QGridLayout()
        self.custom_filter_layout.addLayout(self.kernel_input_grid)

        # Apply Custom Filter Button
        self.apply_custom_filter_button = QPushButton("Apply Custom Filter")
        self.apply_custom_filter_button.clicked.connect(self.apply_custom_filter)
        self.apply_custom_filter_button.clicked.connect(self.update_histogram)
        self.apply_custom_filter_button.clicked.connect(self.update_projection)
        self.custom_filter_layout.addWidget(self.apply_custom_filter_button)

        self.custom_filter_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Initialize kernel input fields
        self.update_kernel_input_grid()

        self.augumentation_label = QLabel("Augmentation")
        self.augumentation_label.setStyleSheet("font-weight: bold;")
        self.custom_filter_layout.addWidget(self.augumentation_label)

        # Rotation section
        self.rotation_layout = QHBoxLayout()

        # Angle input box (SpinBox)
        self.angle_input = QSpinBox()
        self.angle_input.setRange(-180, 180)  # Allow rotation in both directions
        self.angle_input.setValue(0)  # Default to 0 degrees
        self.rotation_layout.addWidget(QLabel("Rotation Angle:"))
        self.rotation_layout.addWidget(self.angle_input)

        # Apply Rotation Button
        self.rotate_button = QPushButton("Rotate")
        self.rotate_button.clicked.connect(self.apply_rotation)
        self.rotate_button.clicked.connect(self.update_histogram)
        self.rotate_button.clicked.connect(self.update_projection)
        self.rotate_button.setStyleSheet("background-color: #d5006d; color: white;") 
        self.rotation_layout.addWidget(self.rotate_button)
        #add under custom filter
        self.custom_filter_layout.addLayout(self.rotation_layout)

        # Create flip buttons
        self.flip_button_layout = QHBoxLayout()

        # Horizontal flip button
        self.horizontal_flip_button = QPushButton("Flip Horizontal")
        self.horizontal_flip_button.clicked.connect(self.apply_flip_horizontal)
        self.horizontal_flip_button.clicked.connect(self.update_histogram)
        self.horizontal_flip_button.clicked.connect(self.update_projection)
        self.horizontal_flip_button.setStyleSheet("background-color: #d5006d; color: white;")  # Light pink with black text
        self.flip_button_layout.addWidget(self.horizontal_flip_button)

        # Vertical flip button
        self.vertical_flip_button = QPushButton("Flip Vertical")
        self.vertical_flip_button.clicked.connect(self.apply_flip_vertical)
        self.vertical_flip_button.clicked.connect(self.update_histogram)
        self.vertical_flip_button.clicked.connect(self.update_projection)
        self.vertical_flip_button.setStyleSheet("background-color: #d5006d; color: white;")  # Light pink with black text
        self.flip_button_layout.addWidget(self.vertical_flip_button)

        # Add the flip buttons layout to the main layout
        self.custom_filter_layout.addLayout(self.flip_button_layout)

        # Checkboxes for histogram
        self.red_checkbox = QCheckBox('Red')
        self.green_checkbox = QCheckBox('Green')
        self.blue_checkbox = QCheckBox('Blue')
        self.mean_checkbox = QCheckBox('Mean')
        
        # Set default checkboxes (enable all colors by default)
        self.red_checkbox.setChecked(True)
        self.green_checkbox.setChecked(True)
        self.blue_checkbox.setChecked(True)
        self.mean_checkbox.setChecked(False)

        # Connect the checkboxes' stateChanged signal to the update_histogram method
        self.red_checkbox.stateChanged.connect(self.update_histogram)
        self.green_checkbox.stateChanged.connect(self.update_histogram)
        self.blue_checkbox.stateChanged.connect(self.update_histogram)
        self.mean_checkbox.stateChanged.connect(self.update_histogram)
        
        # Create a horizontal layout for the checkboxes
        checkbox_layout = QHBoxLayout()
        self.histogram_label = QLabel("Histogram")
        checkbox_layout.addWidget(self.histogram_label)
        checkbox_layout.addWidget(self.red_checkbox)
        checkbox_layout.addWidget(self.green_checkbox)
        checkbox_layout.addWidget(self.blue_checkbox)
        checkbox_layout.addWidget(self.mean_checkbox)

        # Histogram layout
        self.histogram_layout = QVBoxLayout()
        self.histogram_label.setStyleSheet("font-weight: bold;")
        self.canvas_histogram = MplCanvas(self, width=7, height=6, dpi=100)
        self.histogram_layout.addWidget(self.canvas_histogram)

        self.custom_filter_layout.addLayout(checkbox_layout)
        self.custom_filter_layout.addLayout(self.histogram_layout)
        

        # Projection layout
        self.projection_layout = QVBoxLayout()
        self.projection_label = QLabel("Projection")
        self.projection_label.setStyleSheet("font-weight: bold;")
        self.projection_layout.addWidget(self.projection_label)
        self.projection_type = QComboBox(self)
        self.projection_type.addItems(["Horizontal", "Vertical"])
        self.projection_type.setCurrentIndex(0)  
        self.canvas_projection = MplCanvas(self, width=7, height=6, dpi=100)
        self.projection_layout.addWidget(self.projection_type)
        self.projection_layout.addWidget(self.canvas_projection)
        self.projection_type.currentIndexChanged.connect(self.update_projection)
        self.custom_filter_layout.addLayout(self.projection_layout)

        # Buttons layout
        self.button_layout2 = QVBoxLayout()

        # Apply changes button
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_changes)
        self.apply_button.clicked.connect(self.update_histogram)
        self.apply_button.clicked.connect(self.update_projection)
        self.button_layout2.addWidget(self.apply_button)
    
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)
        self.reset_button.clicked.connect(self.update_histogram)
        self.apply_button.clicked.connect(self.update_projection)
        self.button_layout2.addWidget(self.reset_button)

        # Layouts
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.slider_layout)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.button_layout_slow)
        self.main_layout.addLayout(self.button_layout_edge)
        self.main_layout.addLayout(self.morphological_operations_layout)
        self.main_layout.addLayout(self.button_layout2)

        self.big_layout.addLayout(self.main_layout)
        self.big_layout.addLayout(self.custom_filter_layout)

        self.setLayout(self.big_layout)

    def create_menu(self):
        menu_bar = QMenuBar(self)

        # Load menu
        load_menu = menu_bar.addMenu("Load")
        load_action = QAction("Load Image", self)
        load_action.triggered.connect(self.load_image)
        load_action.triggered.connect(self.update_histogram)
        load_action.triggered.connect(self.update_projection)
        load_menu.addAction(load_action)

        # Save menu
        save_menu = menu_bar.addMenu("Save")
        save_jpg_action = QAction("Save as JPG", self)
        save_jpg_action.triggered.connect(lambda: self.save_image("jpg"))
        save_menu.addAction(save_jpg_action)

        save_png_action = QAction("Save as PNG", self)
        save_png_action.triggered.connect(lambda: self.save_image("png"))
        save_menu.addAction(save_png_action)

        save_bmp_action = QAction("Save as BMP", self)
        save_bmp_action.triggered.connect(lambda: self.save_image("bmp"))
        save_menu.addAction(save_bmp_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.instance().quit)
        save_menu.addAction(exit_action)

        # Edit menu with Undo/Redo
        edit_menu = menu_bar.addMenu("Edit")

        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")  # Add keyboard shortcut
        self.undo_action.triggered.connect(self.undo)
        self.undo_action.setEnabled(False)  # Initially disabled
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")  # Add keyboard shortcut
        self.redo_action.triggered.connect(self.redo)
        self.redo_action.setEnabled(False)  # Initially disabled
        edit_menu.addAction(self.redo_action)

        self.big_layout.setMenuBar(menu_bar)


    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif)")
        if file_path:
            self.image_path = file_path
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.original_image = img.copy()  # Store original image
            self.image_processor = ImageProcessor(img)

            self.display_image(self.original_image, self.original_label)
            self.display_image(self.image_processor.image, self.processed_label)


            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(0)  # Reset contrast slider to default


    def display_image(self, image, label):
        """
        Display the image on the label in the GUI.
        """
        if image is None:
            return

        height, width = image.shape[:2]
        if len(image.shape) == 2:  # Grayscale image (2D)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
            height, width = image.shape[:2]

        bytes_per_line = width * 3  # Assuming 3 channels (RGB)
        image_data = image.tobytes()

        q_img = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)

        pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def update_histogram(self):
        if self.image_processor:
            # Determine the image to work with: processed or original
            if self.processed_image is not None:
                processed_image = self.processed_image
            else:
                processed_image = self.image_processor.image

            # Check if the image is a color image (3D) or grayscale (2D)
            if len(processed_image.shape) == 3:  # Color image (height x width x channels)
                r = processed_image[:, :, 0].flatten()  # Red channel
                g = processed_image[:, :, 1].flatten()  # Green channel
                b = processed_image[:, :, 2].flatten()  # Blue channel
            else:  # Grayscale image (height x width)
                r = g = b = processed_image.flatten()  # Use the same data for all channels

            # Combine the channels into one array for the "Mean" histogram
            combined = np.concatenate((r, g, b), axis=0)
            mean = np.mean(combined)

            # Clear the previous histogram plot
            self.canvas_histogram.axes.cla()

            # Plot selected color channels based on checkbox states
            if self.red_checkbox.isChecked():
                self.canvas_histogram.axes.hist(r, bins=256, range=(0,255), color='r', alpha=0.4, label='Red')
            if self.green_checkbox.isChecked():
                self.canvas_histogram.axes.hist(g, bins=256, range=(0,255), color='g', alpha=0.4, label='Green')
            if self.blue_checkbox.isChecked():
                self.canvas_histogram.axes.hist(b, bins=256, range=(0,255), color='b', alpha=0.4, label='Blue')
            if self.mean_checkbox.isChecked():
                self.canvas_histogram.axes.hist(combined, bins=256, range=(0,255), color='gray', alpha=0.4, label='Mean')

            # Customize the histogram appearance
            self.canvas_histogram.axes.tick_params(axis='both', labelsize=6)
            self.canvas_histogram.axes.set_title("RGB Histogram", fontsize=8)
            self.canvas_histogram.axes.set_xlabel("Pixel Value", fontsize=7)
            self.canvas_histogram.axes.set_ylabel("Number of Pixels", fontsize=7)
            self.canvas_histogram.axes.set_xlim(0, 255)

            # Show the histogram axis spines
            self.canvas_histogram.axes.spines['left'].set_visible(True)
            self.canvas_histogram.axes.spines['bottom'].set_visible(True)

            # Make layout adjustments and update the plot
            self.canvas_histogram.figure.tight_layout()
            self.canvas_histogram.draw()



    def apply_brightness(self):
        """Change brightness of the image based on slider value."""
        if self.image_processor:
            brightness_value = self.brightness_slider.value()
            self.processed_image = self.image_processor.adjust_brightness(brightness_value)
            self.display_image(self.processed_image, self.processed_label)

    def apply_contrast(self):
        """Change contrast of the image based on slider value."""
        if self.image_processor:
            contrast_value = self.contrast_slider.value()
            self.processed_image = self.image_processor.adjust_contrast(contrast_value)
            self.display_image(self.processed_image, self.processed_label)

    def convert_to_gray(self):
        """Convert the image to grayscale."""
        if self.image_processor:
            gray_image = self.image_processor.convert_to_gray()
            self.display_image(gray_image, self.processed_label)
            self.processed_image = gray_image

    def negative(self):
        """Convert the image to its negative."""
        if self.image_processor:
            negative_image = self.image_processor.negative()
            self.display_image(negative_image, self.processed_label)
            self.processed_image = negative_image

    def binarization(self):
        """Apply binarization based on slider value."""
        if self.image_processor:
            threshold_value = self.binarization_slider.value()
            self.processed_image = self.image_processor.binarization(threshold_value)
            self.display_image(self.processed_image, self.processed_label)

    def apply_average_filter(self):
        """Apply average filter to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.average_filter()
            self.display_image(self.processed_image, self.processed_label)

    def apply_gaussian_filter(self):
        """Apply Gaussian filter to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.gaussian_filter()
            self.display_image(self.processed_image, self.processed_label)

    def apply_sharpen(self):
        """Apply sharpening filter to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.sharpen()
            self.display_image(self.processed_image, self.processed_label)

    def apply_edge_detection(self, method):
        """Apply edge detection to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.edge_detection(method=method)
            self.display_image(self.processed_image, self.processed_label)
    
    def erosion(self):
        """Apply erosion to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.convolute_binary(self.image_processor.image, self.binary_kernel,'erosion')
            self.display_image(self.processed_image, self.processed_label)
    
    def dilation(self):
        """Apply dilation to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.convolute_binary(self.image_processor.image, self.binary_kernel,'dilation')
            self.display_image(self.processed_image, self.processed_label)

    def opening(self):
        """Apply opening to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.opening(self.binary_kernel)
            self.display_image(self.processed_image, self.processed_label)

    def closing(self):
        """Apply closing to the image."""     
        if self.image_processor:
            self.processed_image = self.image_processor.closing(self.binary_kernel)
            self.display_image(self.processed_image, self.processed_label)


    def update_kernel_input_grid(self):
        """Update the grid layout based on the selected kernel size."""
        # Clear existing input fields
        for i in range(self.kernel_input_grid.count()):
            widget = self.kernel_input_grid.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        kernel_size = int(self.kernel_size_combo.currentText())
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Create QLineEdit for each kernel cell
                weight_input = QLineEdit(self)
                weight_input.setPlaceholderText("0")  # Default placeholder
                #input zeros into weights where values are missing
                weight_input.setText("0")
                self.kernel_input_grid.addWidget(weight_input, i, j)

    def update_kernel_shape(self):
        self.binary_kernel = self.kernel_shape_combo.currentText()

    def apply_custom_filter(self):
        if self.image_processor is None:
            return
        """Apply the custom filter based on user input."""
        kernel_size = int(self.kernel_size_combo.currentText())
        weights = []

        # Retrieve weights from the grid
        for i in range(kernel_size):
            for j in range(kernel_size):
                weight_input = self.kernel_input_grid.itemAt(i * kernel_size + j).widget()
                try:
                    weight = float(weight_input.text())
                    weights.append(weight)
                except ValueError:
                    QMessageBox.warning(self, "Input Error", f"Invalid weight at position ({i}, {j}). Please enter a valid number.")
                    return

        if len(weights) != kernel_size ** 2:
            QMessageBox.warning(self, "Input Error", "Number of weights must match the kernel size.")
            return

        # Convert the list of weights into a numpy array
        weights_array = np.array(weights).reshape((kernel_size, kernel_size))

        # Apply custom filter
        self.processed_image = self.image_processor.apply_custom_filter(weights_array)
        self.display_image(self.processed_image, self.processed_label)

    def apply_rotation(self):
        """
        Apply rotation to the image based on user-selected angle.
        """
        angle = self.angle_input.value()  # Get angle from input
        if self.image_processor is None:
            return
        rotated_image = self.image_processor.rotate(angle)  # Rotate using custom function
        self.processed_image = rotated_image  # Store rotated image as processed image

        # Display the rotated image in the processed label
        self.display_image(self.processed_image, self.processed_label)

    def apply_flip_horizontal(self):
        """Flip the image horizontally and display it."""
        if self.image_processor:
            flipped_image = self.image_processor.flip('horizontal')  # Flip horizontally
            self.processed_image = flipped_image  # Store the flipped image as the processed image

            # Display the flipped image in the processed label
            self.display_image(self.processed_image, self.processed_label)

    def apply_flip_vertical(self):
        """Flip the image vertically and display it."""
        if self.image_processor:
            flipped_image = self.image_processor.flip('vertical')  # Flip vertically
            self.processed_image = flipped_image  # Store the flipped image as the processed image

            # Display the flipped image in the processed label
            self.display_image(self.processed_image, self.processed_label)

    
    def apply_changes(self):
        """Apply the current modifications to the image, allowing further edits."""
        if self.image_processor and self.processed_image is not None:
            self.previous_image = self.image_processor.image.copy()  # Store previous image
            self.image_processor.image = self.processed_image.copy()
            self.display_image(self.image_processor.image, self.original_label)

            self.undo_action.setEnabled(True)  # Enable undo
            self.next_image = None  # Reset redo when applying new change
            self.redo_action.setEnabled(False)  # Disable redo

            self.reset_slider_values()
    
    def undo(self):
        """Revert to the previous image."""
        if self.previous_image is not None:
            self.next_image = self.image_processor.image.copy()  # Store current as next image
            self.image_processor.image = self.previous_image.copy()
            self.processed_image = self.previous_image.copy()

            self.display_image(self.image_processor.image, self.processed_label)
            self.display_image(self.image_processor.image, self.original_label)

            self.redo_action.setEnabled(True)  # Enable redo
            self.undo_action.setEnabled(False)  # Only one undo step

    def redo(self):
        """Reapply the undone change."""
        if self.next_image is not None:
            self.previous_image = self.image_processor.image.copy()  # Store current as previous
            self.image_processor.image = self.next_image.copy()
            self.processed_image = self.next_image.copy()

            self.display_image(self.image_processor.image, self.processed_label)
            self.display_image(self.image_processor.image, self.original_label)

            self.undo_action.setEnabled(True)  # Enable undo
            self.redo_action.setEnabled(False)  # Only one redo step


    def reset_image(self):
        """Reset the image to the original version."""
        if hasattr(self, 'original_image'):
            self.image_processor.image = self.original_image.copy()
            self.processed_image = self.original_image.copy()

            self.display_image(self.original_image, self.processed_label)
            self.display_image(self.original_image, self.original_label)
            self.reset_slider_values()

    def reset_slider_values(self):
        """Reset the slider values."""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.binarization_slider.setValue(128)

    def save_image(self, format):
        """Save the processed image in the selected format."""
        if self.processed_image is None:
            QMessageBox.warning(self, "No Image", "Please process an image before saving.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save Image as {format.upper()}", "", f"Images (*.{format})")
        if file_path:
            if not file_path.endswith(f".{format}"):
                file_path += f".{format}"  # Ensure the file has the correct extension
            
            cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))


    def update_projection(self):
        if self.image_processor:
            if self.processed_image is not None:
                image_gray = np.mean(self.processed_image, axis=2) if self.processed_image.ndim == 3 else self.processed_image
                projection_h = np.sum(image_gray, axis=1)
                projection_v = np.sum(image_gray, axis=0)

            else:
                image_gray = np.mean(self.image_processor.image, axis=2) if self.image_processor.image.ndim == 3 else self.image_processor.image
                projection_h = np.sum(image_gray, axis=1)
                projection_v = np.sum(image_gray, axis=0)

            selected_projection = self.projection_type.currentText()

            self.canvas_projection.axes.cla()


            if selected_projection == "Horizontal":
                self.canvas_projection.axes.hist(np.arange(len(projection_h)), alpha=0.4, bins=len(projection_h),  weights=projection_h, color='b', label='Horizontal')
                self.canvas_projection.axes.set_xlabel("Pixel row", fontsize=7)
                self.canvas_projection.axes.set_ylabel("Number of pixels", fontsize=7)
                self.canvas_projection.axes.set_title("Horizontal Projection", fontsize=8)
            elif selected_projection == "Vertical":
                self.canvas_projection.axes.hist(np.arange(len(projection_v)), alpha=0.4, bins=len(projection_v),  weights=projection_v, color='r', label='Vertical')
                self.canvas_projection.axes.set_xlabel("Pixel column", fontsize=7)
                self.canvas_projection.axes.set_ylabel("Number of pixels", fontsize=7)
                self.canvas_projection.axes.set_title("Vertical Projection", fontsize=8)

            self.canvas_projection.axes.tick_params(axis='both', labelsize=6)
            self.canvas_projection.figure.tight_layout()
            self.canvas_projection.axes.spines['left'].set_visible(True) 
            self.canvas_projection.axes.spines['bottom'].set_visible(True)

            self.canvas_projection.draw()

        

if __name__ == "__main__":
    app = QApplication(sys.argv)

    welcome_dialog = WelcomeDialog()
    if welcome_dialog.exec():  # If user clicks "Get Started", open main GUI
        window = ImageProcessorGUI()
        window.show()
        sys.exit(app.exec())

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import sys
import cv2
import numpy as np
from image_processor import ImageProcessor 
from welcome import WelcomeDialog

class ImageProcessorGUI(QWidget):  
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bildverarbeitung - Image Processing")
        self.setGeometry(100, 100, 700, 500)

        self.image_path = None  
        self.image_processor = None  
        self.processed_image = None  

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
        self.original_label.setFixedSize(300, 300)
        self.original_label.setStyleSheet("border: 2px solid black;")

        # Processed image
        self.processed_label = QLabel("Modified Image")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setFixedSize(300, 300)
        self.processed_label.setStyleSheet("border: 2px solid black;")

        # Add labels to layout
        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.processed_label)

        # Sliders layout
        self.slider_layout = QVBoxLayout()
        
        # Brightness slider
        self.brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.apply_brightness)

        # Contrast slider
        self.contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setMinimum(10)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.apply_contrast)

        # Binarization slider
        self.binarization_label = QLabel("Binarization Threshold:")
        self.binarization_slider = QSlider(Qt.Orientation.Horizontal)
        self.binarization_slider.setMinimum(0)
        self.binarization_slider.setMaximum(255)
        self.binarization_slider.setValue(128)  # Default value
        self.binarization_slider.valueChanged.connect(self.binarization)

        # Add sliders to the layout
        self.slider_layout.addWidget(self.brightness_label)
        self.slider_layout.addWidget(self.brightness_slider)
        self.slider_layout.addWidget(self.contrast_label)
        self.slider_layout.addWidget(self.contrast_slider)
        self.slider_layout.addWidget(self.binarization_label)
        self.slider_layout.addWidget(self.binarization_slider)

        # Buttons layout
        self.button_layout = QVBoxLayout()

        # Fast filter buttons
        self.gray_button = QPushButton("Convert to Gray")
        self.gray_button.clicked.connect(self.convert_to_gray)
        self.gray_button.setStyleSheet("background-color: lightpink; color: black;")  # Light pink with black text
        self.button_layout.addWidget(self.gray_button)

        self.negative_button = QPushButton("Negative")
        self.negative_button.clicked.connect(self.negative)
        self.negative_button.setStyleSheet("background-color: lightpink; color: black;")  # Light pink with black text
        self.button_layout.addWidget(self.negative_button)

        self.button_layout_slow = QHBoxLayout()

        # Slow filter buttons
        self.average_filter_button = QPushButton("Average Filter")
        self.average_filter_button.clicked.connect(self.apply_average_filter)
        self.average_filter_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_slow.addWidget(self.average_filter_button)

        self.gaussian_filter_button = QPushButton("Gaussian Filter")
        self.gaussian_filter_button.clicked.connect(self.apply_gaussian_filter)
        self.gaussian_filter_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_slow.addWidget(self.gaussian_filter_button)

        self.sharpen_button = QPushButton("Sharpen")
        self.sharpen_button.clicked.connect(self.apply_sharpen)
        self.sharpen_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_slow.addWidget(self.sharpen_button)

        self.edge_detection_button = QPushButton("Edge Detection")
        self.edge_detection_button.clicked.connect(self.apply_edge_detection)
        self.edge_detection_button.setStyleSheet("background-color: #d5006d; color: white;")  # Darker pink with white text
        self.button_layout_slow.addWidget(self.edge_detection_button)

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
        self.rotate_button.setStyleSheet("background-color: #d5006d; color: white;") 
        self.rotation_layout.addWidget(self.rotate_button)
        #add under custom filter
        self.custom_filter_layout.addLayout(self.rotation_layout)

        # Create flip buttons
        self.flip_button_layout = QVBoxLayout()

        # Horizontal flip button
        self.horizontal_flip_button = QPushButton("Flip Horizontal")
        self.horizontal_flip_button.clicked.connect(self.apply_flip_horizontal)
        self.horizontal_flip_button.setStyleSheet("background-color: #d5006d; color: white;")  # Light pink with black text
        self.flip_button_layout.addWidget(self.horizontal_flip_button)

        # Vertical flip button
        self.vertical_flip_button = QPushButton("Flip Vertical")
        self.vertical_flip_button.clicked.connect(self.apply_flip_vertical)
        self.vertical_flip_button.setStyleSheet("background-color: #d5006d; color: white;")  # Light pink with black text
        self.flip_button_layout.addWidget(self.vertical_flip_button)

        # Add the flip buttons layout to the main layout
        self.custom_filter_layout.addLayout(self.flip_button_layout)

        # Buttons layout
        self.button_layout2 = QVBoxLayout()

        # Apply changes button
        self.apply_button = QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_changes)
        self.button_layout2.addWidget(self.apply_button)
    
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)
        self.button_layout2.addWidget(self.reset_button)

        # Layouts
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.slider_layout)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.button_layout_slow)
        
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
            self.contrast_slider.setValue(100)  # Reset contrast slider to default


    def display_image(self, image, label):
        """
        Display the image on the label in the GUI.
        """
        if image is None:
            return

        # Get image dimensions
        height, width = image.shape[:2]

        # Ensure the image is in RGB format (3 channels)
        if len(image.shape) == 2:  # Grayscale image (2D)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
            height, width = image.shape[:2]
        
        # Convert the image to a byte array for QImage
        bytes_per_line = width * 3  # Assuming 3 channels (RGB)
        image_data = image.tobytes()

        # Create QImage from image data
        q_img = QImage(image_data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Create QPixmap from QImage
        pixmap = QPixmap.fromImage(q_img)

        # Scale the pixmap to fit inside the label while maintaining the aspect ratio
        pixmap = pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        # Set the QPixmap to the label
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    def apply_brightness(self):
        """Change brightness of the image based on slider value."""
        if self.image_processor:
            brightness_value = self.brightness_slider.value()
            self.processed_image = self.image_processor.adjust_brightness(brightness_value)
            self.display_image(self.processed_image, self.processed_label)

    def apply_contrast(self):
        """Change contrast of the image based on slider value."""
        if self.image_processor:
            contrast_value = self.contrast_slider.value()/100
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

    def apply_edge_detection(self):
        """Apply edge detection to the image."""
        if self.image_processor:
            self.processed_image = self.image_processor.edge_detection()
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
        self.contrast_slider.setValue(100)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)

    welcome_dialog = WelcomeDialog()
    if welcome_dialog.exec():  # If user clicks "Get Started", open main GUI
        window = ImageProcessorGUI()
        window.show()
        sys.exit(app.exec())

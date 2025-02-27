from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import sys
import cv2
import numpy as np
from image_processor import ImageProcessor 

class ImageProcessorGUI(QWidget):  
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bildverarbeitung - Image Processing")
        self.setGeometry(100, 100, 700, 500)

        self.image_path = None  
        self.image_processor = None  
        self.processed_image = None  # Store the processed image for saving

        # Main layout
        self.main_layout = QVBoxLayout()

        # Menu bar
        self.create_menu()

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

        # Store previous and next image for undo/redo
        self.previous_image = None
        self.next_image = None

        # Brightness slider
        self.slider_layout = QVBoxLayout()
        self.brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.apply_brightness)

        self.slider_layout.addWidget(self.brightness_label)
        self.slider_layout.addWidget(self.brightness_slider)

        #Contrast slider
        self.contrast_label = QLabel("Contrast:")
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setMinimum(10)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setTickInterval(1)
        self.contrast_slider.valueChanged.connect(self.apply_contrast)

        self.slider_layout.addWidget(self.contrast_label)
        self.slider_layout.addWidget(self.contrast_slider)

        # Buttons
        self.button_layout = QVBoxLayout()

        # Convert to gray button
        self.gray_button = QPushButton("Convert to Gray")
        self.gray_button.clicked.connect(self.convert_to_gray)
        self.gray_button.setStyleSheet("background-color: #FF1493;") 
        self.button_layout.addWidget(self.gray_button)

        # Negative button
        self.negative_button = QPushButton("Negative")
        self.negative_button.clicked.connect(self.negative)
        self.negative_button.setStyleSheet("background-color: #FF1493;")  
        self.button_layout.addWidget(self.negative_button)

        # Binarization button
        self.binarization_button = QPushButton("Binarization")
        self.binarization_button.clicked.connect(self.binarization)
        self.binarization_button.setStyleSheet("background-color: #FF1493;")  
        self.button_layout.addWidget(self.binarization_button)


        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_changes)
        self.button_layout.addWidget(self.apply_button)

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)

        self.button_layout.addWidget(self.reset_button)

        # Layouts
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.slider_layout)
        self.main_layout.addLayout(self.button_layout)

        self.setLayout(self.main_layout)

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

        self.main_layout.setMenuBar(menu_bar)


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
        """Convert numpy image to QPixmap and display it in QLabel."""
        if image is None:
            return
        height, width, channel = image.shape
        bytes_per_line = channel * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)

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
        """Convert the image to binary."""
        if self.image_processor:
            binary_image = self.image_processor.binarization()
            self.display_image(binary_image, self.processed_label)
            self.processed_image = binary_image
    
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
    window = ImageProcessorGUI()
    window.show()
    sys.exit(app.exec())

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import sys
import cv2
import numpy as np

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bildverarbeitung - Brightness Filter")
        self.setGeometry(100, 100, 700, 500)

        self.image_path = None  # Stores the original image path
        self.original_pixmap = None  # Stores the original image
        self.processed_pixmap = None  # Stores the processed image

        # Main Layout
        self.main_layout = QVBoxLayout()

        # Menu Bar
        self.create_menu()

        # Image Display Layout (Side-by-Side)
        self.image_layout = QHBoxLayout()

        # Original Image Label
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setFixedSize(300, 300)
        self.original_label.setStyleSheet("border: 2px solid black;")

        # Processed Image Label
        self.processed_label = QLabel("Modified Image")
        self.processed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_label.setFixedSize(300, 300)
        self.processed_label.setStyleSheet("border: 2px solid black;")

        # Add Image Labels to Layout
        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.processed_label)

        # Brightness Slider
        self.slider_layout = QHBoxLayout()
        self.brightness_label = QLabel("Brightness:")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setMinimum(-100)  # Min Brightness
        self.brightness_slider.setMaximum(100)  # Max Brightness
        self.brightness_slider.setValue(0)  # Default
        self.brightness_slider.valueChanged.connect(self.apply_brightness)

        self.slider_layout.addWidget(self.brightness_label)
        self.slider_layout.addWidget(self.brightness_slider)

        # Button Layout
        self.button_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_image)

        self.button_layout.addWidget(self.load_button)
        self.button_layout.addWidget(self.reset_button)

        # Add Layouts to Main Layout
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.slider_layout)
        self.main_layout.addLayout(self.button_layout)

        self.setLayout(self.main_layout)

    def create_menu(self):
        menu_bar = QMenuBar(self)
        
        file_menu = menu_bar.addMenu("File")
        
        open_action = QAction("Open Image", self)
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(QApplication.instance().quit)
        file_menu.addAction(exit_action)

        self.main_layout.setMenuBar(menu_bar)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif)")
        if file_path:
            self.image_path = file_path
            self.original_pixmap = QPixmap(file_path)
            self.processed_pixmap = self.original_pixmap.copy()

            self.display_image(self.original_pixmap, self.original_label)
            self.display_image(self.processed_pixmap, self.processed_label)

            self.brightness_slider.setValue(0)  # Reset slider

    def display_image(self, pixmap, label):
        """ Display the given pixmap in the given QLabel. """
        scaled_pixmap = pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)

    def apply_brightness(self):
        """ Adjust the brightness based on the slider value. """
        if self.image_path:
            brightness_value = self.brightness_slider.value()
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply brightness adjustment
            img = np.clip(img + brightness_value, 0, 255).astype(np.uint8)

            # Convert back to QPixmap
            height, width, channel = img.shape
            bytes_per_line = channel * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.processed_pixmap = QPixmap.fromImage(q_img)

            self.display_image(self.processed_pixmap, self.processed_label)

    def reset_image(self):
        """ Reset the image to the original state. """
        if self.original_pixmap:
            self.processed_pixmap = self.original_pixmap.copy()
            self.display_image(self.processed_pixmap, self.processed_label)
            self.brightness_slider.setValue(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessor()
    window.show()
    sys.exit(app.exec())


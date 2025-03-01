from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

class WelcomeDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Welcome to Image Processing App")
        self.setGeometry(200, 200, 600, 500)

        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("Welcome to Image Processing App")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        # Description Label
        description_label = QLabel("Enhance, filter, and transform your images easily! \n\n"
                                   "Features include:\n"
                                   "- Brightness & Contrast Adjustment\n"
                                   "- Grayscale & Negative Conversion\n"
                                   "- Various Image Filters (Gaussian, Sharpen, Edge Detection)\n"
                                   "- Custom Kernel Filtering")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)


        # OK Button to Close
        ok_button = QPushButton("Get Started")
        ok_button.clicked.connect(self.accept)
        ok_button.setStyleSheet("font-size: 16px; font-weight: bold; color: white; background-color: #d5006d; padding: 10px; border-radius: 5px;")

        # Adding Widgets to Layout  
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(ok_button)

        self.setLayout(layout)

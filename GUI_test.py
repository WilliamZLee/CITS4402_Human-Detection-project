from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sys
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import load
import pandas as pd


class HumanDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Human Detection GUI")
        self.resize(900, 600)

        self.image_dir = Path("Testing Images")
        self.model_path = Path("Others/model_final.pkl")
        self.img_size = (128, 64)
        self.model = load(self.model_path)

        self.image_list = sorted(self.image_dir.glob("*.png"))
        self.total_images = len(self.image_list)
        self.current_index = 0
        self.predictions = {}

        self.init_ui()
        self.load_image()

    def init_ui(self):
        # Left: Image Display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(600, 400)

        # Right: Prediction and Buttons
        self.pred_label = QLabel("Prediction: ")
        self.pred_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.pred_label.setAlignment(Qt.AlignCenter)

        self.metric_label = QLabel("Accuracy: --  Precision: --  Recall: --")
        self.metric_label.setAlignment(Qt.AlignCenter)
        self.metric_label.setFont(QFont("Arial", 12))

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.show_previous)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.show_next)

        self.export_button = QPushButton("Export to Excel")
        self.export_button.clicked.connect(self.export_results)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.export_button)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.pred_label)
        right_layout.addWidget(self.metric_label)
        right_layout.addStretch()
        right_layout.addLayout(button_layout)

        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)

    def load_image(self):
        if 0 <= self.current_index < self.total_images:
            img_path = self.image_list[self.current_index]
            pixmap = QPixmap(str(img_path))
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

            # Prediction
            img = imread(str(img_path), as_gray=True)
            resized = resize(img, self.img_size)
            features = hog(resized, orientations=18, pixels_per_cell=(8, 8),
                           cells_per_block=(4, 4), block_norm='L2')
            pred = int(not self.model.predict([features])[0])
            label = "Human" if pred == 1 else "Non-Human"
            self.pred_label.setText(f"Prediction: {label}")

            filename = img_path.name
            self.predictions[filename] = pred

    def show_next(self):
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self.load_image()

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def export_results(self):
        self.load_image()  # Refresh last image

        df = pd.DataFrame(list(self.predictions.items()), columns=["Filename", "Prediction"])
        df.to_excel("predictions.xlsx", index=False)

        label_path = self.image_dir / "labels.txt"
        if label_path.exists():
            true_labels = {}
            with open(label_path, "r") as f:
                next(f)
                for line in f:
                    fname, label = line.strip().split()
                    true_labels[fname] = int(label)

            y_true = []
            y_pred = []
            for fname, pred in self.predictions.items():
                if fname in true_labels:
                    y_true.append(true_labels[fname])
                    y_pred.append(pred)

            if y_true:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                self.metric_label.setText(f"Accuracy: {acc:.2f}  Precision: {prec:.2f}  Recall: {rec:.2f}")
                self.pred_label.setText("Exported predictions.xlsx")
            else:
                self.pred_label.setText("Saved (No ground truth found)")
        else:
            self.pred_label.setText("Saved (No labels.txt found)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HumanDetectionApp()
    window.show()
    sys.exit(app.exec_())



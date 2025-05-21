import os
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.metrics import accuracy_score, precision_score, recall_score
from joblib import load

# === set HOG parameters for test ===
img_size = (128, 64)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# === load model ===
model_path = Path("Others/model.pkl")
if not model_path.exists():
    raise FileNotFoundError("can't find loaded model.pkl, please check if model exist")
clf = load(model_path)
print("model loaded")

# === load test images and get HOG features ===
test_base = Path("data/test")
classes = ['pos', 'neg']
X_test = []
y_test = []

for label, class_name in enumerate(classes):
    folder = test_base / class_name
    for img_file in sorted(folder.glob("*.pgm")):
        img = imread(img_file, as_gray=True)
        img_resized = resize(img, img_size)
        features = hog(img_resized, **hog_params)
        X_test.append(features)
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# === evaluate output ===
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n The output of model performance on test set are:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

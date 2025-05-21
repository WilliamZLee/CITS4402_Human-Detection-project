import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Final model parameters
img_size = (128, 64)
hog_params = {
    'orientations': 18,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (4, 4),
    'block_norm': 'L2'
}

# Paths
model_path = Path("Others/model_final.pkl")
test_path = Path("data/test")
classes = ['pos', 'neg']

# Load model
if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

clf = load(model_path)
print("[Final Evaluation] Model loaded.")

# Prepare test data
X_test = []
y_test = []

for label, class_name in enumerate(classes):
    folder = test_path / class_name
    if not folder.exists():
        raise FileNotFoundError(f"Test folder missing: {folder}")
    for img_file in sorted(folder.glob("*.pgm")):
        img = imread(img_file, as_gray=True)
        img_resized = resize(img, img_size)
        features = hog(img_resized, **hog_params)
        X_test.append(features)
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Predict and evaluate
y_pred = clf.predict(X_test)

print("\n=== Final Model Evaluation ===")
print("Accuracy:  ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall:    ", recall_score(y_test, y_pred))

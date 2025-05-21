import os
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import dump

# set default parameters for HOG and image size
img_size = (128, 64)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# data path
base_path = Path("data/train")
classes = ['pos', 'neg']
output_dir = Path("Others")
output_dir.mkdir(exist_ok=True)

# feature path to store proocessed HOG features
X_npy_path = output_dir / "X_train.npy"
y_npy_path = output_dir / "y_train.npy"

# ==== check if features exist ====
if X_npy_path.exists() and y_npy_path.exists():
    print("Loading, found feature files...")
    X_train = np.load(X_npy_path)
    y_train = np.load(y_npy_path)
else:
    print("Can't get feature files, initiating HOG features processing...")
    X_train = []
    y_train = []
    for label, class_name in enumerate(classes):
        folder = base_path / class_name
        for img_file in sorted(folder.glob("*.pgm")):
            img = imread(img_file, as_gray=True)
            img_resized = resize(img, img_size)
            features = hog(img_resized, **hog_params)
            X_train.append(features)
            y_train.append(label)
            if len(X_train) % 100 == 0:
                print(f"Processed {len(X_train)} images")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    np.save(X_npy_path, X_train)
    np.save(y_npy_path, y_train)

# output
print("Features processed！")
print("X_train shape:", X_train.shape)
print("y_train[:10]:", y_train[:10])
print("feature demenssion:", len(X_train[0]))

# === train SVM ===
print("start training LinearSVC model...")
clf = LinearSVC()
clf.fit(X_train, y_train)

# === output accuracy on train set ===
y_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_pred)
print(f"Model finished, the accuracy on train set is: {train_acc:.4f}")

# === save model ===
model_path = output_dir / "model.pkl"
dump(clf, model_path)
print(f"Model been stored at: {model_path}")
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import dump
from sklearn.svm import LinearSVC

# === Final Parameters ===
img_size = (128, 64)
hog_params = {
    'orientations': 18,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (4, 4),
    'block_norm': 'L2'
}

# === Paths ===
train_path = Path("data/train")
output_dir = Path("Others")
output_dir.mkdir(exist_ok=True)
X_cache = output_dir / "X_train_final.npy"
y_cache = output_dir / "y_train_final.npy"
model_path = output_dir / "model_final.pkl"

# === Feature Extraction ===
if X_cache.exists() and y_cache.exists():
    print("[Final] Loading cached features...")
    X_train = np.load(X_cache)
    y_train = np.load(y_cache)
else:
    print("[Final] Extracting HOG features...")
    classes = ['pos', 'neg']
    X_train = []
    y_train = []

    for label, class_name in enumerate(classes):
        folder = train_path / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        for img_file in sorted(folder.glob("*.pgm")):
            img = imread(img_file, as_gray=True)
            img_resized = resize(img, img_size)
            features = hog(img_resized, **hog_params)
            X_train.append(features)
            y_train.append(label)
            if len(X_train) % 100 == 0:
                print(f"[Final] Processed {len(X_train)} images")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if X_train.ndim != 2:
        raise ValueError(f"[ERROR] Invalid HOG feature shape: {X_train.shape}")
    np.save(X_cache, X_train)
    np.save(y_cache, y_train)

# === Train Model ===
print("[Final] Training Linear SVM...")
clf = LinearSVC()
clf.fit(X_train, y_train)

# === Save Model ===
dump(clf, model_path)
print(f"[Final] Model saved to: {model_path}")
print(f"[Final] X_train shape: {X_train.shape}, y_train sample: {y_train[:10]}")

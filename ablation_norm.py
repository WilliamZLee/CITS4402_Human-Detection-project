import os
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import dump
from sklearn.svm import LinearSVC

# Image size and HOG base params
img_size = (128, 64)
base_path = Path("data/train")
output_dir = Path("Others")
output_dir.mkdir(exist_ok=True)

# Define normalization methods to test (excluding L2-Hys)
norm_list = ["L2", "L1", "L1-sqrt"]

for norm in norm_list:
    print(f"\n=== Running ablation with block_norm = {norm} ===")

    hog_params = {
        'orientations': 9,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'block_norm': norm
    }

    model_path = output_dir / f"model_norm_{norm}.pkl"
    X_cache = output_dir / f"X_train_norm_{norm}.npy"
    y_cache = output_dir / f"y_train_norm_{norm}.npy"

    if X_cache.exists() and y_cache.exists():
        print(f"[{norm}] Loading cached features...")
        X_train = np.load(X_cache)
        y_train = np.load(y_cache)
    else:
        print(f"[{norm}] Extracting HOG features...")
        X_train = []
        y_train = []
        classes = ['pos', 'neg']

        for label, class_name in enumerate(classes):
            folder = base_path / class_name
            for img_file in sorted(folder.glob("*.pgm")):
                img = imread(img_file, as_gray=True)
                img_resized = resize(img, img_size)
                features = hog(img_resized, **hog_params)
                X_train.append(features)
                y_train.append(label)
                if len(X_train) % 100 == 0:
                    print(f"[{norm}] Processed {len(X_train)} images")

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        np.save(X_cache, X_train)
        np.save(y_cache, y_train)

    print(f"[{norm}] Training SVM...")
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    dump(clf, model_path)
    print(f"[{norm}] Model saved to: {model_path}")

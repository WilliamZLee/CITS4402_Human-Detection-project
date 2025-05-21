import os
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import dump
from sklearn.svm import LinearSVC

# Fixed image size
img_size = (128, 64)
base_path = Path("data/train")
output_dir = Path("Others")
output_dir.mkdir(exist_ok=True)

# List of orientation bins to test (excluding 9 = baseline)
bins_list = [3, 4, 6, 8, 12, 15, 18]

for bins in bins_list:
    print(f"\n==== Processing BINS = {bins} ====")

    # Define HOG params dynamically
    hog_params = {
        'orientations': bins,
        'pixels_per_cell': (8, 8),
        'cells_per_block': (2, 2),
        'block_norm': 'L2-Hys'
    }

    # Cache paths for features and model
    X_cache = output_dir / f"X_train_bins{bins}.npy"
    y_cache = output_dir / f"y_train_bins{bins}.npy"
    model_path = output_dir / f"model_bins{bins}.pkl"

    # Check cache
    if X_cache.exists() and y_cache.exists():
        print(f"[{bins}] Loading cached features...")
        X_train = np.load(X_cache)
        y_train = np.load(y_cache)
    else:
        print(f"[{bins}] Extracting HOG features...")
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
                    print(f"[{bins}] Processed {len(X_train)} images")

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        np.save(X_cache, X_train)
        np.save(y_cache, y_train)

    print(f"[{bins}] Training SVM...")
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    dump(clf, model_path)
    print(f"[{bins}] Model saved: {model_path}")

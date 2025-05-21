import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import dump
from sklearn.svm import LinearSVC

# Define block size variants (in number of cells)
block_list = [(2, 2), (3, 3), (4, 4)]  # Corresponds to 16x16, 24x24, 32x32 blocks

# Fixed HOG settings (except block size)
img_size = (128, 64)
orientations = 9
pixels_per_cell = (8, 8)
block_norm = 'L2-Hys'

# Paths
base_path = Path("data/train")
output_dir = Path("Others")
output_dir.mkdir(exist_ok=True)

# Start ablation loop
for block_cfg in block_list:
    block_str = f"{block_cfg[0]}x{block_cfg[1]}"
    print(f"\n==== Training with block size = {block_str} cells ====")

    # Setup HOG config
    hog_params = {
        'orientations': orientations,
        'pixels_per_cell': pixels_per_cell,
        'cells_per_block': block_cfg,
        'block_norm': block_norm
    }

    # Paths for model and cache
    X_cache = output_dir / f"X_train_block_{block_str}.npy"
    y_cache = output_dir / f"y_train_block_{block_str}.npy"
    model_path = output_dir / f"model_block_{block_str}.pkl"

    # Load or compute features
    if X_cache.exists() and y_cache.exists():
        print(f"[{block_str}] Loading cached features...")
        X_train = np.load(X_cache)
        y_train = np.load(y_cache)
    else:
        print(f"[{block_str}] Extracting HOG features...")
        classes = ['pos', 'neg']
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
                    print(f"[{block_str}] Processed {len(X_train)} images")

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        np.save(X_cache, X_train)
        np.save(y_cache, y_train)

    print(f"[{block_str}] Training LinearSVC...")
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    dump(clf, model_path)
    print(f"[{block_str}] Model saved to: {model_path}")

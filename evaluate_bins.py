import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Automatically import bins_list from ablation_bins.py
from ablation_bins import bins_list

# Shared HOG settings
img_size = (128, 64)
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'

# Paths
test_base = Path("data/test")
classes = ['pos', 'neg']
output_dir = Path("Others")

results = []

# Loop through all bin settings
for bins in bins_list:
    print(f"\nEvaluating model with {bins} bins...")

    # Model path
    model_path = output_dir / f"model_bins{bins}.pkl"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        continue

    clf = load(model_path)

    # HOG config
    hog_params = {
        'orientations': bins,
        'pixels_per_cell': pixels_per_cell,
        'cells_per_block': cells_per_block,
        'block_norm': block_norm
    }

    # Extract HOG features from test images
    X_test, y_test = [], []
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

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results.append((bins, acc, prec, rec))

# Print evaluation summary
print("\n=== Evaluation Summary ===")
print(f"{'Bins':<6} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
for bins, acc, prec, rec in results:
    print(f"{bins:<6} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f}")


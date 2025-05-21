import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Import the block size list from ablation script
from ablation_blocksize import block_list

# Fixed HOG parameters (except for block size)
img_size = (128, 64)
orientations = 9
pixels_per_cell = (8, 8)
block_norm = 'L2-Hys'

# Paths
test_base = Path("data/test")
output_dir = Path("Others")
classes = ['pos', 'neg']

# Store evaluation results
results = []

# Loop over each block size
for block_cfg in block_list:
    block_str = f"{block_cfg[0]}x{block_cfg[1]}"
    print(f"\nEvaluating model with block size = {block_str} cells...")

    # Setup HOG parameters
    hog_params = {
        'orientations': orientations,
        'pixels_per_cell': pixels_per_cell,
        'cells_per_block': block_cfg,
        'block_norm': block_norm
    }

    # Load model
    model_path = output_dir / f"model_block_{block_str}.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        continue
    clf = load(model_path)

    # Extract HOG features for test set
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

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results.append((block_str, acc, prec, rec))

# Print evaluation table
print("\n=== Block Size Evaluation Summary ===")
print(f"{'Block':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
for block_str, acc, prec, rec in results:
    print(f"{block_str:<8} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f}")

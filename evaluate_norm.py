import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Only evaluating the ablation variants (not including baseline L2-Hys)
norm_list = ["L2", "L1", "L1-sqrt"]

# Fixed HOG parameters (except block_norm)
img_size = (128, 64)
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# Paths
test_base = Path("data/test")
output_dir = Path("Others")
classes = ['pos', 'neg']

results = []

for norm in norm_list:
    print(f"\nEvaluating model with block_norm = {norm}...")

    model_path = output_dir / f"model_norm_{norm}.pkl"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        continue

    clf = load(model_path)

    hog_params = {
        'orientations': orientations,
        'pixels_per_cell': pixels_per_cell,
        'cells_per_block': cells_per_block,
        'block_norm': norm
    }

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

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results.append((norm, acc, prec, rec))

# Print result table
print("\n=== Normalization Method Evaluation Summary ===")
print(f"{'Norm':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
for norm, acc, prec, rec in results:
    print(f"{norm:<10} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f}")

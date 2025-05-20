import os
import numpy as np
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import dump

# 设置图像尺寸和 HOG 参数
img_size = (128, 64)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# 数据路径和保存路径
base_path = Path("data/train")
classes = ['pos', 'neg']
output_dir = Path("Others")
output_dir.mkdir(exist_ok=True)

# 特征文件路径
X_npy_path = output_dir / "X_train.npy"
y_npy_path = output_dir / "y_train.npy"

# ==== 判断缓存是否存在 ====
if X_npy_path.exists() and y_npy_path.exists():
    print("检测到缓存特征文件，正在加载...")
    X_train = np.load(X_npy_path)
    y_train = np.load(y_npy_path)
else:
    print("未检测到缓存，开始提取 HOG 特征...")
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

# 输出信息
print("特征准备完成！")
print("X_train shape:", X_train.shape)
print("y_train[:10]:", y_train[:10])
print("特征维度:", len(X_train[0]))

# === 训练 SVM ===
print("开始训练 LinearSVC 模型...")
clf = LinearSVC()
clf.fit(X_train, y_train)

# === 模型训练完成，输出训练准确率 ===
y_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train, y_pred)
print(f"模型训练完成，训练集准确率: {train_acc:.4f}")

# === 保存模型 ===
model_path = output_dir / "model.pkl"
dump(clf, model_path)
print(f"模型已保存至: {model_path}")
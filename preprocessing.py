import os
import shutil
from pathlib import Path

# === 配置 ===
base_path = Path("DC-ped-dataset_base")
output_path = Path("data")
train_pos_target = output_path / "train" / "pos"
train_neg_target = output_path / "train" / "neg"
test_pos_target = output_path / "test" / "pos"
test_neg_target = output_path / "test" / "neg"

# 创建目标目录
for folder in [train_pos_target, train_neg_target, test_pos_target, test_neg_target]:
    folder.mkdir(parents=True, exist_ok=True)

# === 定义复制函数 ===
def copy_images(src_dirs, subfolder, dst_dir, max_count, prefix):
    count = 0
    for src_dir in src_dirs:
        files = sorted((base_path / src_dir / subfolder).glob("*.pgm"))
        for file in files:
            if count >= max_count:
                return
            new_name = f"{prefix}_{count+1:05d}.pgm"
            shutil.copy(file, dst_dir / new_name)
            count += 1

# === 执行复制 ===
# 训练集：从 1 和 2 中各取 500 张正样本和负样本
copy_images(["1", "2"], "ped_examples", train_pos_target, 1000, "pos")
copy_images(["1", "2"], "non-ped_examples", train_neg_target, 1000, "neg")

# 测试集：从 T1 中各取 100 张
copy_images(["T1"], "ped_examples", test_pos_target, 100, "pos")
copy_images(["T1"], "non-ped_examples", test_neg_target, 100, "neg")

import pandas as pd

# 可视化输出目录结构（前5张每类）
result = []
for split in ["train", "test"]:
    for label in ["pos", "neg"]:
        folder = output_path / split / label
        images = sorted(folder.glob("*.pgm"))[:5]
        for img in images:
            result.append({
                "Set": split,
                "Class": label,
                "Filename": img.name,
                "Path": str(img.relative_to(output_path))
            })

df = pd.DataFrame(result)
print(df.head())
df.to_csv("dataset_overview.csv", index=False)


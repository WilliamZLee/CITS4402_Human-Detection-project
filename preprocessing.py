import os
import shutil
from pathlib import Path

# === config and file path initialize ===
base_path = Path("DC-ped-dataset_base")
output_path = Path("data")
train_pos_target = output_path / "train" / "pos"
train_neg_target = output_path / "train" / "neg"
test_pos_target = output_path / "test" / "pos"
test_neg_target = output_path / "test" / "neg"

# create target file path
for folder in [train_pos_target, train_neg_target, test_pos_target, test_neg_target]:
    folder.mkdir(parents=True, exist_ok=True)

# === define image copy function ===
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

# === select images ===
# train set 1000 neg & 1000 pos
copy_images(["1", "2"], "ped_examples", train_pos_target, 1000, "pos")
copy_images(["1", "2"], "non-ped_examples", train_neg_target, 1000, "neg")

# test set from T1 1000neg 100 pos
copy_images(["T1"], "ped_examples", test_pos_target, 100, "pos")
copy_images(["T1"], "non-ped_examples", test_neg_target, 100, "neg")

import pandas as pd

# output top 5 on the processed list
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


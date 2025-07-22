import os
import json
import shutil
from tqdm import tqdm
import random

# === Paths ===
json_path = r"D:\ODD PROJECT\DATA\BDD\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"
image_root = r"D:\ODD PROJECT\DATA\BDD\bdd100k\bdd100k\images\100k\train"
id_target = r"D:\ODD PROJECT\DATA\BDD\scene_classification\id"
ood_target = r"D:\ODD PROJECT\DATA\BDD\scene_classification\ood\unknown"

# === Parameters ===
num_id = 500
num_ood = 500

# === Create target folders ===
os.makedirs(id_target, exist_ok=True)
os.makedirs(ood_target, exist_ok=True)

# === Load JSON metadata ===
with open(json_path, 'r') as f:
    data = json.load(f)

# === Filter filenames by time of day and file existence ===
all_files = set(os.listdir(image_root))
day_files = [item['name'] for item in data if item['attributes']['timeofday'] == 'daytime' and item['name'] in all_files]
night_files = [item['name'] for item in data if item['attributes']['timeofday'] == 'night' and item['name'] in all_files]

print(f"Found {len(day_files)} valid daytime images.")
print(f"Found {len(night_files)} valid nighttime images.")

# === Shuffle and select subset ===
random.shuffle(day_files)
random.shuffle(night_files)

selected_day = day_files[:num_id]
selected_night = night_files[:num_ood]

# === Copy ID images ===
print(f"Copying {len(selected_day)} ID (day) images...")
for fname in tqdm(selected_day):
    src = os.path.join(image_root, fname)
    dst = os.path.join(id_target, fname)
    shutil.copy(src, dst)

# === Copy OOD images ===
print(f"Copying {len(selected_night)} OOD (night) images...")
for fname in tqdm(selected_night):
    src = os.path.join(image_root, fname)
    dst = os.path.join(ood_target, fname)
    shutil.copy(src, dst)

print("✅ Done: Large-scale dataset created for ID and OOD.")

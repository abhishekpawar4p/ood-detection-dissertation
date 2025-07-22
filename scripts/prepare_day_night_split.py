import os
import json
import shutil
from tqdm import tqdm

# ---- CONFIG ----
source_image_dir = r"D:\ODD PROJECT\DATA\BDD\bdd100k\bdd100k\images\100k\train"
json_label_path = r"D:\ODD PROJECT\DATA\BDD\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json"

output_id_dir = r"D:\ODD PROJECT\DATA\BDD\scene_classification\ID"
output_ood_dir = r"D:\\ODD PROJECT\DATA\BDD\scene_classification\OOD"

num_per_class = 500

# ---- CREATE FOLDERS IF NEEDED ----
os.makedirs(output_id_dir, exist_ok=True)
os.makedirs(output_ood_dir, exist_ok=True)

# ---- LOAD LABELS ----
with open(json_label_path, "r") as f:
    labels = json.load(f)

day_images = []
night_images = []

# ---- FILTER BY TIME OF DAY ----
for item in labels:
    if "attributes" in item and "timeofday" in item["attributes"]:
        time = item["attributes"]["timeofday"]
        name = item["name"]
        if time == "daytime":
            day_images.append(name)
        elif time == "night":
            night_images.append(name)

print(f"Found {len(day_images)} daytime images, {len(night_images)} nighttime images.")

# ---- COPY IMAGES ----
for img_name in tqdm(day_images[:num_per_class], desc="Copying ID (day)"):
    src = os.path.join(source_image_dir, img_name)
    dst = os.path.join(output_id_dir, img_name)
    if os.path.exists(src):
        shutil.copy(src, dst)

for img_name in tqdm(night_images[:num_per_class], desc="Copying OOD (night)"):
    src = os.path.join(source_image_dir, img_name)
    dst = os.path.join(output_ood_dir, img_name)
    if os.path.exists(src):
        shutil.copy(src, dst)

print(" Image splitting complete!")

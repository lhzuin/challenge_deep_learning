import os
import random
import csv
from PIL import Image, ImageEnhance, ImageFilter
import shutil
from tqdm import tqdm

input_images_dir = "dataset/train_val_gpt"  # dossier contenant les images d'origine
output_images_dir = "dataset/train_val_gpt_aug"  # dossier de sortie pour les images augmentées
descriptions_csv = "dataset/train_val_gpt.csv"  # fichier CSV : image,description
output_csv = "dataset"  # dossier de sortie pour le CSV

os.makedirs(output_images_dir, exist_ok=True)

descriptions = {}


img_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith('.jpg')]

# Read descriptions from CSV
with open(descriptions_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        descriptions[row['id']] = row

def perturb_image(img):
    angle = random.uniform(-10, 10)
    img = img.rotate(angle)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    pixels = img.load()
    for _ in range(int(img.size[0] * img.size[1] * 0.01)):
        x = random.randint(0, img.size[0] - 1)
        y = random.randint(0, img.size[1] - 1)
        pixels[x, y] = tuple(random.randint(0, 255) for _ in range(3))
    return img

n_augmentations = 3
output_csv = os.path.join(output_csv, "train_val_gpt_aug.csv")

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Unnamed: 0', 'id', 'channel', 'title', 'date', 'views', 'year', 'summary']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in tqdm(img_files, desc="Images"):
        img_path = os.path.join(input_images_dir, filename)
        id = filename[:-4]
        if not os.path.isfile(img_path):
            print(f"Image manquante : {filename}")
            continue

        shutil.copy(img_path, os.path.join(output_images_dir, filename))
        writer.writerow(descriptions[id])
        img = Image.open(img_path).convert("RGB")

        for i in tqdm(range(n_augmentations), desc=f"Aug {filename}", leave=False):
            img_aug = perturb_image(img)
            new_name = f"{id}_aug{i+1}.jpg"
            new_path = os.path.join(output_images_dir, new_name)
            img_aug.save(new_path)
            d = descriptions[id].copy()
            d['id'] = new_name
            writer.writerow(d)
print("Augmentation terminée.")
import os
import random
import csv
from PIL import Image, ImageEnhance, ImageFilter
import shutil
from tqdm import tqdm
from datetime import datetime, timedelta
import random
from torchvision import transforms as T

input_images_dir = "dataset/train_val"  # dossier contenant les images d'origine
output_images_dir = "dataset/train_val_gpt_aug3"  # dossier de sortie pour les images augmentées
descriptions_csv = "dataset/train_val.csv"  # fichier CSV : image,description
descriptions_csv2 = "dataset/train_val_gpt2.csv"  # fichier CSV : image,description
descriptions_csv3 = "dataset/train_val_gpt3.csv"  # fichier CSV : image,description
descriptions_csv4 = "dataset/train_val_gpt4.csv"  # fichier CSV : image,description
output_csv = "dataset"  # dossier de sortie pour le CSV
name_out_csv = "train_val_gpt_aug3.csv"

os.makedirs(output_images_dir, exist_ok=True)

descriptions:dict = {}
descriptions2:dict = {}
descriptions3:dict = {}
descriptions4:dict = {}
descriptions_lst = [descriptions, descriptions2, descriptions3, descriptions4]


img_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith('.jpg')]

# Read descriptions from CSV
with open(descriptions_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        descriptions[row['id']] = row
        descriptions[row['id']]['aug']=False

with open(descriptions_csv2, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row["title"] = row.pop("new_title")
        descriptions_lst[1][row['id']] = row
with open(descriptions_csv3, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row["title"] = row.pop("new_title")
        descriptions_lst[2][row['id']] = row

with open(descriptions_csv4, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        row["title"] = row.pop("new_title")
        descriptions_lst[3][row['id']] = row

aug_pipeline = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=Image.BICUBIC),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
])

def perturb_image(img):
    img = aug_pipeline(img)
    angle = random.uniform(-4, 4)
    img = img.rotate(angle)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    pixels = img.load()
    for _ in range(int(img.size[0] * img.size[1] * 0.008)):
        x = random.randint(0, img.size[0] - 1)
        y = random.randint(0, img.size[1] - 1)
        pixels[x, y] = tuple(random.randint(0, 255) for _ in range(3))
    return img

def jitter_timestamp(ts: str, max_days: int = 2) -> str:
    """
    Shift an ISO-format timestamp (e.g. "2024-11-25 17:47:19+00:00")
    by a random integer in [-max_days, +max_days] days,
    preserving the original time and timezone offset.
    """
    # Python 3.7+ can parse the "+00:00" offset
    dt = datetime.fromisoformat(ts)
    shift = random.randint(-max_days, max_days)
    dt_jittered = dt + timedelta(days=shift)
    return dt_jittered.isoformat()

n_augmentations = 3
output_csv = os.path.join(output_csv, name_out_csv)

with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Unnamed: 0', 'id', 'channel', 'title', 'date', 'views', 'year', 'summary',"aug"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in tqdm(img_files, desc="Images"):
        img_path = os.path.join(input_images_dir, filename)
        id = filename[:-4]
        if not os.path.isfile(img_path):
            print(f"Image manquante : {filename}")
            continue

        shutil.copy(img_path, os.path.join(output_images_dir, filename))
        writer.writerow(descriptions_lst[0][id])
        img = Image.open(img_path).convert("RGB")

        for i in range(1,n_augmentations+1):
            img_aug = perturb_image(img)
            new_name = f"{id}_aug{i+1}.jpg"
            new_path = os.path.join(output_images_dir, new_name)
            img_aug.save(new_path)
            d = descriptions_lst[i][id].copy()
            d['id'] = new_name[:-4]
            d['aug']=True
            d["date"] = jitter_timestamp(d["date"])
            writer.writerow(d)
print("Augmentation terminée.")
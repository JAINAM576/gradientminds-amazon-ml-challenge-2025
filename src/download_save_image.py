import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from io import BytesIO
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder
from sentence_transformers import SentenceTransformer
import joblib
from tqdm import tqdm
import os

df1=pd.read_csv("train_image.csv")

image_feature = 'image_link' 
sample_id_col = 'sample_id'
def download_and_save_images(df, image_dir="images"):
    os.makedirs(image_dir, exist_ok=True)
    saved_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        img_url = row[image_feature]
        sample_id = str(row[sample_id_col])
        ext = os.path.splitext(img_url)[-1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"
        local_path = os.path.join(image_dir, f"{sample_id}{ext}")

        if not os.path.exists(local_path):
            try:
                response = requests.get(img_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(local_path)
            except Exception as e:
                print(f"Warning: Could not download {img_url}: {e}")
                image = Image.new("RGB", (224, 224), color=(255, 255, 255))
                image.save(local_path)

        saved_paths.append(local_path)
    return saved_paths



def download_and_save_images_by_sampleids(df, sample_ids, image_dir="images"):

    os.makedirs(image_dir, exist_ok=True)
    saved_paths = []

    subset_df = df[df[sample_id_col].isin(sample_ids)]

    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Downloading images"):
        img_url = row[image_feature]
        sample_id = str(row[sample_id_col])

        if not isinstance(img_url, str) or not img_url.startswith("http"):
            print(f"Warning: Invalid image link for sample_id={sample_id}")
            continue

        ext = os.path.splitext(img_url)[-1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"

        local_path = os.path.join(image_dir, f"{sample_id}{ext}")

        if not os.path.exists(local_path):
            try:
                response = requests.get(img_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(local_path)
            except Exception as e:
                print(f"Warning: Could not download {img_url} (sample_id={sample_id}): {e}")
                image = Image.new("RGB", (224, 224), color=(255, 255, 255))
                image.save(local_path)

        saved_paths.append(local_path)

    return saved_paths
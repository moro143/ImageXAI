import os

import numpy as np
import pandas as pd

folders = [
    "06_musical instrument",
    "05_insect",
    "03_reptile",
    "04_carnivore",
    "00_dog",
    "08_fish",
    "02_wheeled vehicle",
    "01_bird",
    "07_primate",
]
base_path = f"tests/"
directory_path = f"{base_path}Imagenet/fg_mask/val"

data = []

# Iterate through each folder
for dir in folders:
    full_folder_path = f"{directory_path}/{dir}"
    files = os.listdir(full_folder_path)
    # Iterate through each file in the folder
    for file_name in files:
        mask = np.load(f"{full_folder_path}/{file_name}")
        data.append((file_name, mask.sum() / (mask.shape[0] * mask.shape[1])))

df = pd.DataFrame(data, columns=["file_name", "size_percent"])
df.to_pickle("sizes.pkl")

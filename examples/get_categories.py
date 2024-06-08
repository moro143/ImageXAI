import os

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
directory_path = f"{base_path}Imagenet/original/val"

data = []

# Iterate through each folder
for dir in folders:
    full_folder_path = f"{directory_path}/{dir}"
    files = os.listdir(full_folder_path)
    # Iterate through each file in the folder
    for file_name in files:
        # Append a tuple of the directory and file name to the data list
        data.append((dir, file_name))

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=["Directory", "File Name"])
df.to_pickle("categories.pkl")

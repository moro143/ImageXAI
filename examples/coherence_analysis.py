import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from ImageXAI.measurements import (
    confidence_change_mask,
    confidence_change_wo_mask,
    iou_explanations,
)

base_model = ResNet50(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.output)

results_list = []

exp_types = [["gradcam", "lime"], ["gradcam", "shap"], ["lime", "shap"]]

categories = [
    "00_dog",
    "01_bird",
    "02_wheeled vehicle",
    "03_reptile",
    "04_carnivore",
    "05_insect",
    "06_musical instrument",
    "07_primate",
    "08_fish",
]
base_path = "tests/"
images_done_path = f"{base_path}explanations/images_done"

with open(images_done_path, "r") as f:
    images_done = f.read().splitlines()

c = 0
for img_name in images_done:
    print(c / len(images_done))
    c += 1
    for category in categories:
        path_img = f"{base_path}Imagenet/original/val/{category}/{img_name}"
        path_fg = f"{base_path}/Imagenet/fg_mask/val/{category}/{img_name[:-4]}npy"
        if os.path.exists(path_img):
            break
    img = image.load_img(path_img, target_size=(224, 224))
    img_array_t = image.img_to_array(img)
    img_array = img_array_t.astype(int)
    img_array = tf.expand_dims(img_array, 0)
    img_processed = preprocess_input(img_array)
    mask = np.load(path_fg)

    for exp1_type, exp2_type in exp_types:
        path_explanation1 = f"{base_path}explanations/{img_name}-{exp1_type}.pkl"
        path_explanation2 = f"{base_path}explanations/{img_name}-{exp2_type}.pkl"
        with open(path_explanation1, "rb") as f:
            exp1 = pickle.load(f)
        with open(path_explanation2, "rb") as f:
            exp2 = pickle.load(f)
        results_list.append(
            {
                "img_name": img_name,
                "exp_type1": exp1_type,
                "exp_type2": exp2_type,
                "iou": iou_explanations(
                    exp1=exp1, exp1_type=exp1_type, exp2=exp2, exp2_type=exp2_type
                ),
            }
        )
results_df = pd.DataFrame(results_list)

results_df.to_pickle("coherence_analysis.pkl")

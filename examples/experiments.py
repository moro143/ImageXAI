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
    iou,
    new_predictions,
)

base_model = ResNet50(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.output)

results_list = []

exp_types = ["gradcam", "lime", "shap"]

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

    for exp_type in exp_types:
        path_explanation = f"{base_path}explanations/{img_name}-{exp_type}.pkl"
        with open(path_explanation, "rb") as f:
            exp = pickle.load(f)
        old_pred, new_pred = new_predictions(
            img_processed=img_processed, exp=exp, exp_type=exp_type, model=model
        )
        old_pred_wo_exp, new_pred_wo_exp = new_predictions(
            img_processed=img_processed,
            exp=exp,
            exp_type=exp_type,
            model=model,
            no_explanation=True,
        )
        results_list.append(
            {
                "img_name": img_name,
                "exp_type": exp_type,
                "iou": iou(mask=mask, exp=exp, exp_type=exp_type),
                "old_pred": old_pred,
                "new_pred": new_pred,
                "old_pred_wo_exp": old_pred_wo_exp,
                "new_pred_wo_exp": new_pred_wo_exp,
            }
        )
results_df = pd.DataFrame(results_list)

results_df.to_pickle("experiment.pkl")

import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (ResNet50,
                                                    decode_predictions,
                                                    preprocess_input)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from ImageXAI.explanations import (explain_with_gradcam, explain_with_lime,
                                   explain_with_shap)

base_model = ResNet50(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.output)

base_path = "data/"


def save_explanation(
    explanation, img_name, xai_method, path=f"{base_path}explanations/"
):
    filename = f"{img_name}-{xai_method}.pkl"

    with open(path + filename, "wb") as f:
        pickle.dump(explanation, f)


def load_explanation(filename, path=f"{base_path}explanations/"):
    with open(path + filename, "rb") as f:
        return pickle.load(f)



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
directory_path = f"{base_path}Imagenet/original/val"

images_done_path = f"{base_path}explanations/images_done"

with open(images_done_path, "r") as f:
    images_done = f.read().splitlines()

i = 0
for dir in folders:
    full_folder_path = f"{directory_path}/{dir}"
    files = os.listdir(full_folder_path)
    for file_name in files:
        if file_name not in images_done:
            full_file_path = f"{full_folder_path}/{file_name}"
            img = image.load_img(full_file_path, target_size=(224, 224))
            img_array_t = image.img_to_array(img)
            img_array = img_array_t.astype(int)
            img_array = tf.expand_dims(img_array, 0)
            img_processed = preprocess_input(img_array)
            l_exp = explain_with_lime(model, img_processed)
            save_explanation(l_exp, file_name, "lime")
            s_exp = explain_with_shap(model, img_processed)
            save_explanation(s_exp, file_name, "shap")
            g_exp = explain_with_gradcam(model, img_processed)
            save_explanation(g_exp, file_name, "gradcam")

            with open(images_done_path, "a") as f:
                f.write(file_name + "\n")
    i += 1
    print(dir, i / len(folders))

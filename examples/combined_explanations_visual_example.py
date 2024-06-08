import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image

from ImageXAI.explanations import combine_explanations
from ImageXAI.measurements import (
    confidence_change_mask,
    confidence_change_wo_mask,
    iou,
    new_predictions,
)

name = "n02510455_41143.JPEG"
exp1_path = f"tests/explanations/{name}-gradcam.pkl"
exp2_path = f"tests/explanations/{name}-shap.pkl"
category = "04_carnivore"
image_path = f"tests/Imagenet/original/val/{category}/{name}"

with open(exp1_path, "rb") as f:
    exp1 = pickle.load(f)
with open(exp2_path, "rb") as f:
    exp2 = pickle.load(f)

img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)

result1 = exp1 > 0.2
result1 = result1.astype(float)
result1 = np.repeat(result1, 224 // exp1.shape[0], axis=0)
result1 = np.repeat(result1, 224 // exp1.shape[1], axis=1)
plt.imshow(img / 255 * result1[:, :, np.newaxis])
plt.show()


result = exp2.values[0].squeeze(axis=-1)
result = result > result.mean()  # how to
result = result.astype(float)
result = result.mean(axis=-1)

plt.imshow(img / 255 * result[:, :, np.newaxis])
plt.show()

combine_or = combine_explanations({"gradcam": exp1, "shap": exp2})
plt.imshow(img / 255 * combine_or[:, :, np.newaxis])
plt.show()
combine_and = combine_explanations({"gradcam": exp1, "shap": exp2}, sum=False)
plt.imshow(img / 255 * combine_and[:, :, np.newaxis])
plt.show()

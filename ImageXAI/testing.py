import explanations
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.segmentation import mark_boundaries
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image

model = ResNet50()
img_path = "test.png"
img = image.load_img(img_path, target_size=(224, 224))
img_array_t = image.img_to_array(img)
img_array = img_array_t.astype(int)
img_array = tf.expand_dims(img_array, 0)
img_processed = preprocess_input(img_array)

x = explanations.explain_with_lime(model, img_processed, 1000, 1)
temp, mask = x.get_image_and_mask(x.top_labels[0])
x = explanations.explain_with_shap(model, img_processed)
print(x)
print(temp)
print(mask)
temp /= 255
plt.imshow(mark_boundaries(temp, mask * 255))
plt.show()
print(x)

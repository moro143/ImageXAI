import numpy as np
import shap
import tensorflow as tf
from lime import lime_image
from tensorflow.keras.applications.resnet50 import preprocess_input  # pyright: ignore


def explain_with_lime(model, img, num_samples=1000, top_labels=1):
    explainer = lime_image.LimeImageExplainer()
    img_numpy = img.numpy()
    explanation = explainer.explain_instance(
        img_numpy[0],
        model.predict,
        top_labels=top_labels,
        num_samples=num_samples,
        random_seed=42,
    )
    return explanation


def explain_with_shap(model, img):
    np.random.seed(42)

    def f(X):
        tmp = X.copy()
        preprocess_input(tmp)
        return model(tmp)

    masker = shap.maskers.Image("inpaint_telea", img[0].shape)  # type: ignore
    explainer = shap.Explainer(f, masker)
    shap_values = explainer(
        np.array(img),
        max_evals=500,  # type: ignore
        batch_size=50,  # type: ignore
        outputs=shap.Explanation.argsort.flip[:1],  # type: ignore
    )
    return shap_values


def explain_with_gradcam(model, img, layer="conv5_block3_out"):
    last_conv_layer = model.get_layer(layer)
    classifier_layer = model.layers[-1]
    grad_model = tf.keras.models.Model(
        model.inputs, [last_conv_layer.output, classifier_layer.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        top_k = tf.argsort(predictions[0])[-1:]
        target_class = top_k[-1]
        target_conv_output = conv_output[0]
        loss = predictions[:, target_class]
        grads = tape.gradient(loss, conv_output)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_mean(tf.multiply(weights, target_conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

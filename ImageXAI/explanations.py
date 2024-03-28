import numpy as np
import shap
import tensorflow as tf
from lime import lime_image


def explain_with_lime(model, img, num_samples=1000, top_labels=1):
    explainer = lime_image.LimeImageExplainer()
    # img_numpy = img.numpy()
    img_numpy = img
    explanation = explainer.explain_instance(
        img_numpy,
        model.predict,
        top_labels=top_labels,
        num_samples=num_samples,
    )
    return explanation


def explain_with_shap(model, img):
    def f(X):
        tmp = X.copy()
        return model(tmp)

    masker = shap.maskers.Image("inpaint_relea", img[0].shape)
    explainer = shap.Explainer(f, masker)
    shap_values = explainer(
        img.numpy(),
        max_evals=500,
        batch_size=50,
        outputs=shap.Explanation.argsort.flip[:1],
    )
    return shap_values


def explain_with_gradcam(model, img, layer="conv5_block3_out"):
    last_conv_layer = model.get_layer(layer)
    classifier_layer = model.layers[-1]
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, classifier_layer.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        predicted_class = tf.argmax(predictions[0])
        top_k = tf.argsort(predictions[0])[-1:]
        top_k_values = [predictions[0][i] for i in top_k]
        target_class = top_k[-1]
        target_conv_output = conv_output[0]
        loss = predictions[:, target_class]
    grads = tape.gradient(loss, conv_output)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_mean(tf.multiply(weights, target_conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)  # do i need this
    heatmap /= np.max(heatmap)
    return heatmap

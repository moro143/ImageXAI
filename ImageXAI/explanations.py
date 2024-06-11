import numpy as np
import shap
import tensorflow as tf
from lime import lime_image
from tensorflow.keras.applications.resnet50 import \
    preprocess_input  # pyright: ignore


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
        print(target_class)
        target_conv_output = conv_output[0]
        loss = predictions[:, target_class]
        grads = tape.gradient(loss, conv_output)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_mean(tf.multiply(weights, target_conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


def combine_explanations(exps, threshold=0.2, sum=True):
    mask = np.ones([224, 224])
    if sum:
        mask = np.zeros([224, 224])
    for exp_type, exp in exps.items():
        if exp_type == "gradcam":
            result = exp > threshold
            result = result.astype(float)
            result = np.repeat(result, 224 // exp.shape[0], axis=0)
            result = np.repeat(result, 224 // exp.shape[1], axis=1)
        elif exp_type == "lime":
            _, result = exp.get_image_and_mask(exp.top_labels[0])
        elif exp_type == "shap":
            result = exp.values[0].squeeze(axis=-1)
            result[result<0]=0
            max_val = np.max(result)
            result = result/max_val
            result = result > threshold 
            result = result.astype(float)
            result = result.mean(axis=-1)
        else:
            return
        if sum:
            mask = np.logical_or(mask, result).astype(int)
        else:
            mask = np.logical_and(mask, result).astype(int)
    return mask

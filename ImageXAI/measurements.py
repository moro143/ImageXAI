import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def iou(mask, exp, exp_type, threshold=0.2, img_shape=(224, 224)):
    if exp_type == "gradcam":
        result = exp > threshold
        result = result.astype(float)
        result = np.repeat(result, img_shape[0] // exp.shape[0], axis=0)
        result = np.repeat(result, img_shape[1] // exp.shape[1], axis=1)
    elif exp_type == "lime":
        _, result = exp.get_image_and_mask(exp.top_labels[0])
    elif exp_type == "shap":
        result = exp.values[0].squeeze(axis=-1)
        result = result > result.mean()  # how to
        result = result.astype(float)
        result = result.mean(axis=-1)
    else:
        return

    intersection = np.logical_and(result, mask).sum()
    union = result.sum() + mask.sum() - intersection
    iou = intersection / union

    return iou


def iou_explanations(
    exp1, exp1_type, exp2, exp2_type, threshold=0.2, img_shape=(224, 224)
):
    if exp1_type == "gradcam":
        result1 = exp1 > threshold
        result1 = result1.astype(float)
        result1 = np.repeat(result1, img_shape[0] // exp1.shape[0], axis=0)
        result1 = np.repeat(result1, img_shape[1] // exp1.shape[1], axis=1)
    elif exp1_type == "lime":
        _, result1 = exp1.get_image_and_mask(exp1.top_labels[0])
    elif exp1_type == "shap":
        result1 = exp1.values[0].squeeze(axis=-1)
        result1 = result1 > result1.mean()  # how to
        result1 = result1.astype(float)
        result1 = result1.mean(axis=-1)
    else:
        return
    if exp2_type == "gradcam":
        result2 = exp2 > threshold
        result2 = result2.astype(float)
        result2 = np.repeat(result2, img_shape[0] // exp2.shape[0], axis=0)
        result2 = np.repeat(result2, img_shape[1] // exp2.shape[1], axis=1)
    elif exp2_type == "lime":
        _, result2 = exp2.get_image_and_mask(exp2.top_labels[0])
    elif exp2_type == "shap":
        result2 = exp2.values[0].squeeze(axis=-1)
        result2 = result2 > result2.mean()  # how to
        result2 = result2.astype(float)
        result2 = result2.mean(axis=-1)
    else:
        return

    intersection = np.logical_and(result1, result2).sum()
    union = result1.sum() + result2.sum() - intersection
    iou = intersection / union

    return iou


def confidence_change_mask(
    img_processed, exp, exp_type, model, threshold=0.2, img_shape=(224, 224)
):
    predictions = model.predict(img_processed)
    base_confidence = max(predictions[0])
    prediction_index = np.where(predictions[0] == base_confidence)[0][0]
    if exp_type == "gradcam":
        result = exp > threshold
        result = result.astype(float)
        result = np.repeat(result, img_shape[0] // exp.shape[0], axis=0)
        result = np.repeat(result, img_shape[1] // exp.shape[1], axis=1)
    elif exp_type == "lime":
        _, result = exp.get_image_and_mask(exp.top_labels[0])
    elif exp_type == "shap":
        result = exp.values[0].squeeze(axis=-1)
        result = result > result.mean()  # how to
        result = result.astype(float)
        result = result.mean(axis=-1)
    else:
        return
    inverted_mask = np.where(result == 0, 1, 0)
    cut_image = img_processed[0] * inverted_mask[:, :, np.newaxis]
    cut_image_processed = tf.expand_dims(cut_image, 0)
    new_prediction = model.predict(cut_image_processed)

    return new_prediction[0][prediction_index] - base_confidence


def confidence_change_wo_mask(
    img_processed, exp, exp_type, model, threshold=0.2, img_shape=(224, 224)
):
    predictions = model.predict(img_processed)
    base_confidence = max(predictions[0])
    prediction_index = np.where(predictions[0] == base_confidence)[0][0]
    if exp_type == "gradcam":
        result = exp > threshold
        result = result.astype(float)
        result = np.repeat(result, img_shape[0] // exp.shape[0], axis=0)
        result = np.repeat(result, img_shape[1] // exp.shape[1], axis=1)
    elif exp_type == "lime":
        _, result = exp.get_image_and_mask(exp.top_labels[0])
    elif exp_type == "shap":
        result = exp.values[0].squeeze(axis=-1)
        result = result > result.mean()
        result = result.astype(float)
        result = result.mean(axis=-1)
    else:
        return
    cut_image = img_processed[0] * result[:, :, np.newaxis]
    cut_image_processed = tf.expand_dims(cut_image, 0)

    new_prediction = model.predict(cut_image_processed)

    return new_prediction[0][prediction_index] - base_confidence


def new_predictions(
    img_processed,
    exp,
    exp_type,
    model,
    threshold=0.2,
    img_shape=(224, 224),
    no_explanation=False,
):
    predictions = model.predict(img_processed)
    base_confidence = max(predictions[0])
    prediction_index = np.where(predictions[0] == base_confidence)[0][0]
    if exp_type == "gradcam":
        result = exp > threshold
        result = result.astype(float)
        result = np.repeat(result, img_shape[0] // exp.shape[0], axis=0)
        result = np.repeat(result, img_shape[1] // exp.shape[1], axis=1)
    elif exp_type == "lime":
        _, result = exp.get_image_and_mask(exp.top_labels[0])
    elif exp_type == "shap":
        result = exp.values[0].squeeze(axis=-1)
        result = result > result.mean()
        result = result.astype(float)
        result = result.mean(axis=-1)
    else:
        return None, None
    if no_explanation:
        result = np.where(result == 0, 1, 0)
    cut_image = img_processed[0] * result[:, :, np.newaxis]
    cut_image_processed = tf.expand_dims(cut_image, 0)

    new_prediction = model.predict(cut_image_processed)

    return base_confidence, new_prediction[0][prediction_index]

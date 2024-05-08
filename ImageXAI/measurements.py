import numpy as np
import tensorflow as tf


def iou(mask, exp, exp_type, threshold=0.5):
    if exp_type == "gradcam":
        result = exp > threshold
        result = result.astype(float)
        result = np.repeat(result, mask.shape[0] // exp.shape[0], axis=0)
        result = np.repeat(result, mask.shape[1] // exp.shape[1], axis=1)
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


def confidence_change_mask(img_processed, mask, exp, exp_type, model, threshold=0.5):
    predictions = model.predict(img_processed)
    base_confidence = max(predictions[0])
    prediction_index = np.where(predictions[0] == base_confidence)[0][0]
    if exp_type == "gradcam":
        result = exp > threshold
        result = result.astype(float)
        result = np.repeat(result, mask.shape[0] // exp.shape[0], axis=0)
        result = np.repeat(result, mask.shape[1] // exp.shape[1], axis=1)
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

    new_prediction[0][prediction_index]
    return base_confidence - new_prediction[0][prediction_index]


def confidence_change_wo_mask(img_processed, mask, exp, exp_type, model, threshold=0.5):
    predictions = model.predict(img_processed)
    base_confidence = max(predictions[0])
    prediction_index = np.where(predictions[0] == base_confidence)[0][0]
    if exp_type == "gradcam":
        result = exp > threshold
        result = result.astype(float)
        result = np.repeat(result, mask.shape[0] // exp.shape[0], axis=0)
        result = np.repeat(result, mask.shape[1] // exp.shape[1], axis=1)
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

    new_prediction[0][prediction_index]
    return base_confidence - new_prediction[0][prediction_index]

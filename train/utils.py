import numpy as np


def pred_to_label(classifier_outputs, regressor_outputs):
    result = np.zeros((classifier_outputs.shape[0], 6))
    mask = (classifier_outputs >= 0.5)
    result[mask] = regressor_outputs[mask]
    return result


def pred_to_5_label(classifier_outputs, regressor_outputs):
    result = np.zeros((classifier_outputs.shape[0], 5))
    mask = (classifier_outputs >= 0.5)
    result[mask] = regressor_outputs[mask]
    return result


def get_y(batch, classifier_outputs, regressor_outputs):
    regressor_outputs = regressor_outputs.cpu().numpy()
    regressor_outputs = regressor_outputs.argmax(axis=-1) + 1
    y_true = batch['labels_regressor'].numpy()
    y_pred = np.round(pred_to_label(classifier_outputs.cpu().numpy(), regressor_outputs))
    return y_pred, y_true


def get_y_(batch, classifier_outputs, regressor_outputs):
    regressor_outputs = regressor_outputs.cpu().numpy()
    regressor_outputs = regressor_outputs.argmax(axis=-1) + 1
    y_true = batch['labels_regressor'].numpy()
    y_pred = np.round(pred_to_5_label(classifier_outputs.cpu().numpy(), regressor_outputs))
    return y_pred, y_true

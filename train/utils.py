import numpy as np


def pred_to_label(classifier_outputs, regressor_outputs):
    result = np.zeros((classifier_outputs.shape[0], 6))
    mask = (classifier_outputs >= 0.5)
    result[mask] = regressor_outputs[mask]
    return result


def get_y(batch, classifier_outputs, regressor_outputs):
    outputs_regressor = regressor_outputs.cpu().numpy()
    outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
    y_true = batch['labels_regressor'].numpy()
    y_pred = np.round(pred_to_label(classifier_outputs.cpu().numpy(), regressor_outputs))
    return y_pred, y_true

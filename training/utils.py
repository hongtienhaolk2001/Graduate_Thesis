import numpy as np


def prob_to_label_1(pred):
    mask = (pred >= 0.5)
    x_coor, y_coor = np.where(mask)
    result = np.zeros((pred.shape[0], 6))
    for x, y in zip(x_coor, y_coor):
        loc = y // 6
        star = y % 6
        result[x][loc] = star
    return result


def prob_to_label_2(pred):
    result = np.zeros((pred.shape[0], 6))
    pred = pred.reshape(pred.shape[0], -1, 5)
    star = pred.argmax(axis=-1) + 1
    prob = pred.max(axis=-1)
    mask = prob >= 0.5
    result[mask] = star[mask]
    return result


def pred_to_label(outputs_classifier, outputs_regressor):
    """Convert output model to label. Get aspects have reliability >= 0.5
	Args:
		outputs_classifier (numpy.array): Output classifier layer
		outputs_regressor (numpy.array): Output regressor layer
	Returns:
		predicted label
	"""

    result = np.zeros((outputs_classifier.shape[0], 6))
    mask = (outputs_classifier >= 0.5)
    result[mask] = outputs_regressor[mask]
    return result

import numpy as np
import torch


def pred_to_label(outputs_classifier, outputs_regressor):
    """
    Convert output model to label. Get aspects have reliability >= 0.5.

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


def update_model(model, score, best_score):
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), "weights/model.pt")
        print(f"update model with score {best_score}")
    return best_score

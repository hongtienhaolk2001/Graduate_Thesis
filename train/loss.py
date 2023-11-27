import torch.nn.functional as F
import torch
import torch.nn as nn


def loss_classifier(pred_classifier, labels_classifier):
    """
    Binary Cross Entropy Loss for classification tasks.

    Args:
        pred_classifier (torch.Tensor): Predictions from the classifier.
        labels_classifier (torch.Tensor): True labels for the classification task.

    Returns:
        torch.Tensor: Binary Cross Entropy Loss.
    """
    return nn.BCELoss()(pred_classifier, labels_classifier)


def loss_regressor(pred_regressor, labels_regressor):
    """
    Mean Squared Error Loss for regression tasks with a masking mechanism.

    Args:
        pred_regressor (torch.Tensor): Predictions from the regressor.
        labels_regressor (torch.Tensor): True labels for the regression task.

    Returns:
        torch.Tensor: Mean Squared Error Loss.
    """
    mask = (labels_regressor != 0)
    loss = ((pred_regressor - labels_regressor) ** 2)[mask].sum() / mask.sum()
    return loss


def loss_softmax(inputs, labels, device):
    """
    Cross Entropy Loss for softmax-based multi-class classification tasks.

    Args:
        inputs (torch.Tensor): Model predictions.
        labels (torch.Tensor): True labels for the multi-class classification task.
        device: Device on which the computation should be performed.

    Returns:
        torch.Tensor: Cross Entropy Loss.
    """
    mask = (labels != 0)
    n, aspect, rate = inputs.shape
    num = 0
    loss = torch.zeros(labels.shape).to(device)
    for i in range(aspect):
        label_i = labels[:, i].clone()
        label_i[label_i != 0] -= 1
        label_i = label_i.type(torch.LongTensor).to(device)
        loss[:, i] = nn.CrossEntropyLoss(reduction='none')(inputs[:, i, :], label_i)
    loss = loss[mask].sum() / mask.sum()
    return loss


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2,
                       reduction: str = "none", ):
    """
    Sigmoid Focal Loss for binary classification tasks.

    Args:
        inputs (torch.Tensor): Model predictions before sigmoid activation.
        targets (torch.Tensor): True labels for the binary classification task.
        alpha (float): Weighting factor for class imbalance.
        gamma (float): Focusing parameter to down-weight easy examples.
        reduction (str): Specifies the reduction to apply to the output.

    Returns:
        torch.Tensor: Sigmoid Focal Loss.
    """
    # p = torch.sigmoid(inputs)
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


# def bce_loss_weights(inputs, targets, weights):
#     ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
#     weights = targets * (1 / weights.view(1, -1)) + (1 - targets) * (1 / (1 - weights.view(1, -1)))
#     loss = ce_loss * weights
#     return loss.mean()


# def CB_loss(inputs, targets, samples_positive_per_cls, samples_negative_per_cls, no_of_classes=2, loss_type='sigmoid',
#             beta=0.9999, gamma=2):
#     samples_per_cls = torch.concat([samples_positive_per_cls.unsqueeze(-1),
#                                     samples_negative_per_cls.unsqueeze(-1)],
#                                    dim=-1)  # num_cls, 2
#     effective_num = 1.0 - torch.pow(beta, samples_per_cls)  # num_cls, 2
#     weights = (1.0 - beta) / effective_num  # num_cls, 2
#     weights = weights / weights.sum(dim=-1).reshape(-1, 1) * no_of_classes  # num_cls, 2
#     weights = targets * weights[:, 0] + (1 - targets) * weights[:, 1]
#
#     if loss_type == "focal":
#         cb_loss = (sigmoid_focal_loss(inputs, targets) * weights).mean()
#     elif loss_type == "sigmoid":
#         cb_loss = (F.binary_cross_entropy(inputs, targets, reduction="none") * weights).mean()
#     return cb_loss

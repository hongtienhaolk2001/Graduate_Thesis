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


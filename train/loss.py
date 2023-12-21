import torch.nn.functional as F
import torch
import torch.nn as nn


def binary_CE(outputs, targets):
    return nn.BCELoss()(outputs, targets)


def softmax(predict, labels, device):
    mask = (labels != 0)
    _, topic, _ = predict.shape  # n, topic, aspect
    loss = torch.zeros(labels.shape).to(device)
    for i in range(topic):
        label_i = labels[:, i].clone()
        label_i[label_i != 0] -= 1
        label_i = label_i.type(torch.LongTensor).to(device)
        loss[:, i] = nn.CrossEntropyLoss(reduction='none')(predict[:, i, :], label_i)
    loss = loss[mask].sum() / mask.sum()
    return loss


def sigmoid_focal(inputs: torch.Tensor, targets: torch.Tensor,
                  alpha: float = 0.25, gamma: float = 2,
                  reduction: str = "none", ):
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


def custom_loss_1(batch, outputs_classifier, outputs_regressor, device):
    sigmoid_focal_loss = sigmoid_focal(outputs_classifier, batch['labels_classifier'].to(device).float(),
                                       alpha=-1, gamma=1, reduction='mean')
    softmax_loss = softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)
    return 10 * sigmoid_focal_loss + softmax_loss


def custom_loss_2(batch, outputs_classifier, outputs_regressor, device):
    classifier_loss = binary_CE(outputs_classifier, batch['labels_classifier'].to(device).float())
    softmax_loss = softmax(outputs_regressor, batch['labels_regressor'].to(device).float(), device)
    return classifier_loss + softmax_loss


# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     predict, labels = torch.rand(-1, 6, 4), torch.rand(-1, 6, 4)
    # print(labels != 0)
    # print(predict.clone())
    # print(softmax(predict, labels, device))


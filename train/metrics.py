import numpy as np


class ScalarMetric:
    def __init__(self):
        self.scalar = 0
        self.num = 0

    def update(self, scalar):
        self.scalar += scalar
        self.num += 1
        return self

    def compute(self):
        return self.scalar / self.num

    def reset(self):
        self.scalar = 0
        self.num = 0


def precision(y_pred, y_true):
    TP = np.logical_and(y_pred, y_true).sum(axis=0)
    FP = np.logical_and(y_pred, np.logical_not(y_true)).sum(axis=0)
    return TP / (TP + FP)


def recall(y_pred, y_true):
    TP = np.logical_and(y_pred, y_true).sum(axis=0)
    FN = np.logical_and(np.logical_not(y_pred), y_true).sum(axis=0)
    return TP / (TP + FN)


class F1_score:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def update(self, y_pred, y_true):
        self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0) if self.y_pred is not None else y_pred
        self.y_true = np.concatenate([self.y_true, y_true], axis=0) if self.y_true is not None else y_true

    def compute(self, aggregation=None):
        f1_score = np.zeros(self.y_pred.shape[1])
        precision_score = precision(self.y_pred != 0, self.y_true != 0)
        recall_score = recall(self.y_pred != 0, self.y_true != 0)
        mask_precision_score = np.logical_and(precision_score != 0, np.logical_not(np.isnan(precision_score)))
        mask_recall_score = np.logical_and(recall_score != 0, np.logical_not(np.isnan(recall_score)))
        mask = np.logical_and(mask_precision_score, mask_recall_score)
        f1_score[mask] = 2 * (precision_score[mask] * recall_score[mask]) / (precision_score[mask] + recall_score[mask])
        if aggregation == "macro":
            f1_score = f1_score.mean()
        return f1_score


class R2_score:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def update(self, y_pred, y_true):
        self.y_pred = np.concatenate([self.y_pred, y_pred], axis=0) if self.y_pred is not None else y_pred
        self.y_true = np.concatenate([self.y_true, y_true], axis=0) if self.y_true is not None else y_true

    def compute(self):
        mask = np.logical_and(self.y_pred != 0, self.y_true != 0)
        rss = (((self.y_pred - self.y_true) ** 2) * mask).sum(axis=0)
        k = (mask * 16).sum(axis=0)
        r2_score = np.ones(rss.shape[0])
        mask2 = (k != 0)
        r2_score[mask2] = 1 - rss[mask2] / k[mask2]
        return r2_score

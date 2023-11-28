import matplotlib.pyplot as plt
from metrics import metric


class Visualization:
    def __init__(self):
        self.log = {'loss': [], "Classifier loss": [], "Regressor loss": [],
                    "Accuracy": [], "F1": [], "R2": [], "Score": []}
        self.best_score = -1

    def add2log(self, value: metric):
        self.log['loss'].append(value.loss)
        self.log['Classifier loss'].append(value.classifier_loss)
        self.log['Regressor loss'].append(value.regressor_loss)
        self.log['Accuracy'].append(value.acc)
        F1 = value.f1_score.compute()
        R2 = value.r2_score.compute()
        final_score = (F1 * R2).sum() * 1 / 6
        self.log['F1'].append(F1)
        self.log['R2'].append(R2)
        self.log['Score'].append(final_score)

    def visualize(self):
        pass

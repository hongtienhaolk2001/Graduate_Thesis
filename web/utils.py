from web import app
import pandas as pd
import json
import os
import numpy as np


def read_json(path):
    with open(os.path.join(app.root_path, path), 'r', encoding='utf8') as f:
        return json.load(f)


def load_categories():
    return read_json(r'data/categories.json')


def pred_to_label(outputs_classifier, outputs_regressor):
    result = np.zeros((outputs_classifier.shape[0], 6))
    mask = (outputs_classifier >= 0.5)
    result[mask] = outputs_regressor[mask]
    return result



if __name__ == '__main__':
    # print(load_news(product_id=1, page=1))

    pass

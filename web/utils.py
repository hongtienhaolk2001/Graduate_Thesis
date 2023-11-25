import pandas as pd
from web import app
import json
import os


def read_json(path):
    with open(os.path.join(app.root_path, path), 'r', encoding='utf8') as f:
        return json.load(f)


def load_categories():
    return read_json(r'data/categories.json')


def count_product():
    return 120


def load_news(product_id=None, page=1):
    pass


if __name__ == '__main__':
    print(load_news(product_id=1, page=1))

import numpy as np
from app.models import *


def load_categories():
    return Category.query.all()
    # with open(os.path.join(app.root_path, r'data/categories.json'), 'r', encoding='utf8') as f:
    #     return json.load(f)


def count_news(category_id):
    if category_id != 0:
        return News.query.filter(News.category_id.__eq__(int(category_id))).count()
    return News.query.filter().count()


def load_news(category_id, keyword=None, page=1):
    news = News.query.filter()
    if category_id != 0:
        news = news.filter(News.category_id.__eq__(category_id))
    if keyword:
        news = news.filter(News.brief.contains(keyword))

    page_size = app.config['PAGE_SIZE']
    start = (page - 1) * page_size
    end = start + page_size

    return news.slice(start, end).all()


def pred_to_label(outputs_classifier, outputs_regressor):
    result = np.zeros((outputs_classifier.shape[0], 6))
    mask = (outputs_classifier >= 0.5)
    result[mask] = outputs_regressor[mask]
    return result


if __name__ == '__main__':
    pass

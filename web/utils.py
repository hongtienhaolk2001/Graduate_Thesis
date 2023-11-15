import pandas as pd
from web import *



def count_product():
    return 120


def load_news(product_id=None, page=1):
    news = pd.read_csv(r"../data/raw_data/rice.csv")
    page_size = app.config['PAGE_SIZE']
    start = (page - 1) * page_size
    end = start + page_size

    try:
        df = news[product_id][start:end]
        return {'brief': df['brief'],
                'content': df['content'],
                'date': df['date'],
                'title': df['title'],
                'title': df['category']}
    except:
        df = news[start:end]
        return {'brief': df['brief'],
                'content': df['content'],
                'date': df['date'],
                'title': df['title'],
                'title': df['category']}


if __name__ == '__main__':
    print(load_news(product_id=1, page=1))

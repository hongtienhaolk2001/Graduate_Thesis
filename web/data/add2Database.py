from web.models import *
from web import db, app
import pandas as pd
import hashlib


def add_Category(file_path):
    category = pd.read_csv(file_path)
    for i in range(0, len(category)):
        db.session.add(Category(name=category['name'][i]),
                       category_id=int(category['id'][i],))


def add_News(file_path):
    # Add News Data
    news = pd.read_csv(file_path)
    for i in range(0, len(news)):
        db.session.add(News(category_id=int(news['category'][i]),
                            name=news['title'][i],
                            brief=news['brief'][i],
                            date=news['date'][i],
                            content=news['content'][i], ))


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Add Category
        category = pd.read_csv(r'categories.csv')
        for i in range(0, len(category)):
            c = Category(name=category['name'][i])
            db.session.add(c)
        # Add Books Data
        news = pd.read_csv(r'rice.csv')
        for i in range(0, len(news)):
            db.session.add(News(title=news['title'][i],
                                brief=news['brief'][i],
                                date=news['date'][i],
                                category_id=news['category'][i],
                                content=news['content'][i], ))
        news = pd.read_csv(r'coffee.csv')
        for i in range(0, len(news)):
            db.session.add(News(title=news['title'][i],
                                brief=news['brief'][i],
                                date=news['date'][i],
                                category_id=news['category'][i],
                                content=news['content'][i], ))
        db.session.commit()

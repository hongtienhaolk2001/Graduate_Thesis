from app.models import *
from app import db, app
import pandas as pd
import hashlib


def add_Category(file_path):
    category = pd.read_csv(file_path)
    for i in range(0, len(category)):
        c = Category(name=category['name'][i])
        db.session.add(c)


def add_News(file_path):
    news = pd.read_csv(file_path)
    for i in range(0, len(news)):
        db.session.add(News(title=news['title'][i],
                            brief=news['brief'][i],
                            date=news['date'][i],
                            category_id=news['category'][i],
                            content=news['content'][i], ))





if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        add_Category(r'categories.csv')
        add_News(r'rice.csv')
        add_News(r'coffee.csv')
        db.session.commit()
    print('Add data successfully')

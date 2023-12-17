import numpy as np
from app import app, db
from app.models import News, Category, User
import hashlib


def load_categories():
    return Category.query.all()


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


def add_user(name, username, password, **kwargs):
    """adds a new user to the database.
    """
    password = str(hashlib.md5(password.encode('utf-8')).hexdigest())
    user = User(name=name, username=username, password=password, email=kwargs.get('email'))
    db.session.add(user)
    try:
        db.session.commit()
    except:
        return False
    else:
        return True


def check_login(username, password):
    """checks whether the given username and password match a user in the database.
    """
    if username and password:
        password = str(hashlib.md5(password.strip().encode('utf8')).hexdigest())
        return User.query.filter(User.username.__eq__(username.strip()),
                                 User.password.__eq__(password)).first()


def get_user_by_id(user_id):
    return User.query.get(user_id)

from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship

from app import *
# from main import app



class BaseModel(db.Model):
    __abstract__ = True
    id = Column(Integer, primary_key=True, autoincrement=True)


class Category(BaseModel):
    __tablename__ = 'category'
    name = Column(String(20), nullable=False)
    products = relationship('News', backref='category', lazy=False)

    def __str__(self):
        return self.name


class News(BaseModel):
    __tablename__ = 'news'
    title = Column(Text, nullable=False)
    brief = Column(Text)
    date = Column(Text)
    content = Column(Text)
    category_id = Column(Integer, ForeignKey(Category.id), nullable=False)

    def __str__(self):
        return self.name

# if __name__ == '__main__':
#     pass

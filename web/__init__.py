from flask import Flask
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager
# from model.CustomSoftmaxModel import ModelInference

app = Flask(__name__)
app.secret_key = '1234567890qwertyuiop'




# # Segmenter input
# rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
#                          annotators="wseg",
#                          max_heap_size='-Xmx500m')

# Tokenizer
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True)


# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Tienhao123@localhost/ecommerce?charset=utf8mb4'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# app.config['PAGE_SIZE'] = 10
# db = SQLAlchemy(app=app)
# login_manager = LoginManager(app=app)
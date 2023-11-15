from flask import Flask
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager
# from model.CustomSoftmaxModel import ModelInference

app = Flask(__name__)
app.secret_key = '1234567890qwertyuiop'
app.config['PAGE_SIZE'] = 10
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/ecommerce?charset=utf8mb4'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# db = SQLAlchemy(app=app)
# login_manager = LoginManager(app=app)


# # Segmenter input
# rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
#                          annotators="wseg",
#                          max_heap_size='-Xmx500m')
# Tokenizer
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True)
# Get model
# model = ModelInference(tokenizer, rdrsegmenter, r'./model/weights/model.pt')



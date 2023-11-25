from flask import Flask
from analysis import ModelInference
from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer
# from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager


app = Flask(__name__)
app.secret_key = '1234567890qwertyuiop'
app.config['PAGE_SIZE'] = 10
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/ecommerce?charset=utf8mb4'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# db = SQLAlchemy(app=app)
# login_manager = LoginManager(app=app)
phobert_model = ModelInference(tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True),
                               rdrsegmenter=VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m'),
                               model_path='weights/model.pt')
phobert_model = None




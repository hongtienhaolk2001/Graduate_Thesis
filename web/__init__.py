from flask import Flask
from analysis import ModelInference
from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer
from flask_sqlalchemy import SQLAlchemy
# from flask_login import LoginManager


app = Flask(__name__)
app.secret_key = '1234567890qwertyuiop'
app.config['PAGE_SIZE'] = 10
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/news?charset=utf8mb4'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# db = SQLAlchemy(app=app)
# login_manager = LoginManager(app=app)



rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True)
phobert_model = ModelInference(model_path='weights/model.pt', tokenizer=tokenizer, rdrsegmenter=rdrsegmenter, )




from flask import Flask
from .analysis import ModelInference
from vncorenlp import VnCoreNLP
from transformers import AutoTokenizer
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os
import time


def load_model(root_path):
    start_time = time.time()
    rdrsegmenter = VnCoreNLP(os.path.join(root_path, "vncorenlp/VnCoreNLP-1.1.1.jar"), annotators="wseg",
                             max_heap_size='-Xmx500m')
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True)
    print(f"load model successfully - {round(time.time()-start_time,2)} second")
    return ModelInference(model_path=os.path.join(root_path, "weights/model.pt"), tokenizer=tokenizer,
                          rdrsegmenter=rdrsegmenter, )


def create_app():
    app = Flask(__name__)
    app.secret_key = '1234567890qwertyuiop'
    app.config['PAGE_SIZE'] = 10
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/news?charset=utf8mb4'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    return app


def create_db(app):
    return SQLAlchemy(app=app)


app = create_app()
login_manager = LoginManager(app=app)
root_path = app.root_path
phobert_model = load_model(root_path)
db = create_db(app)



from flask import Flask
import os
from logging import Formatter, INFO
from logging.handlers import RotatingFileHandler

def loadconfig(app):
    app.config.from_pyfile('config.py')

def initlogging(app):
    if not os.path.exists(app.config['LOGDIR']):
        os.mkdir(app.config['LOGDIR'])

    file_handler = RotatingFileHandler(f"{app.config['LOGDIR']}/covid19kg.log", maxBytes=10240,backupCount=10)
    file_handler.setFormatter(Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    app.logger.addHandler(file_handler)
    app.logger.setLevel(INFO)
    app.logger.info('covid19kg startup')

def create_app(app):
    loadconfig(app)
    initlogging(app)
    return app


flask_app = Flask(__name__)
flask_app = create_app(flask_app)
embedding_file = flask_app.config['EMBEDDING_FILE']

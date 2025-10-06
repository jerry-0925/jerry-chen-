import os
class Config:
    SECRET_KEY = os.urandom(24)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'  # Or any other DB URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    FLASK_RUN_PORT=1001

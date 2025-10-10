from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
from .extensions import db
from .config import Config
import pymysql
import os
pymysql.install_as_MySQLdb()


def create_app():
    app = Flask(__name__, template_folder='app/templates')
    app.config.from_object(Config)
    print(f"Flask is searching for templates in: {os.path.abspath(app.template_folder)}")
    # Enable CORS for all routes
    CORS(app)

    # Initialize extensions
    db.init_app(app)

    # Register Blueprints and routes
    from .routes import routes
    app.register_blueprint(routes)

    # Static files serving
    @app.route('/')
    def index():
        template_path = os.path.abspath("app/templates/index.html")

        print(f"Expected template path: {template_path}")
        return render_template("index.html")

    @app.route('/register')
    def register():
        return render_template('register.html')

    @app.route('/login')
    def login():
        return render_template('login.html')

    @app.route('/upload_music')
    def upload_music():
        return render_template('upload_music.html')

    @app.route('/tasks')
    def tasks():
        return render_template('tasks.html')

    from flask import render_template

    return app


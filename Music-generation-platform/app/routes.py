from flask import Blueprint, request, jsonify
from .models import db, User, Music, Task, UserBehavior
import uuid
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
routes = Blueprint('routes', __name__)

@routes.route('/api/register', methods=['POST'])
def register_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if user:
        return jsonify({'code': 401, 'message': 'Email already exists'}), 401

    hashed_password = generate_password_hash(password)
    new_user = User(email=email, password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'code': 200, 'message': 'Success'}), 200

@routes.route('/api/login', methods=['POST'])
def login_user():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'code': 401, 'message': 'Invalid email or password'}), 401

    access_token = str(uuid.uuid4())
    return jsonify({'code': 200, 'access_token': access_token, 'message': 'Success'}), 200

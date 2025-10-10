from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from .models import db, User

auth = Blueprint('auth', __name__)

# Add registration and login logic here (already included in /routes.py)

from .extensions import db
import datetime

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    quota = db.Column(db.Integer, default=100)
    quota_update_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Music Model
class Music(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    artist = db.Column(db.String(50))
    file_path = db.Column(db.String(255))
    play_count = db.Column(db.Integer, default=0)
    valence = db.Column(db.Float)
    arousal = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Task Model
class Task(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    token_id = db.Column(db.String(50))
    prompt = db.Column(db.Text)
    status = db.Column(db.Enum('pending', 'processing', 'completed', 'failed'), default='pending')
    file_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# User Behavior Model
class UserBehavior(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    music_id = db.Column(db.Integer, db.ForeignKey('music.id'))
    rate = db.Column(db.Float)

from flask import Blueprint, request, jsonify
from .models import db, Music, User, UserBehavior
import uuid

music = Blueprint('music', __name__)

@music.route('/api/music/upload/<user_id>', methods=['POST'])
def music_upload(user_id):
    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({'code': 401, 'message': 'User not logged in'}), 401

    data = request.json
    directory = data.get('directory')

    valence = 0.5
    arousal = 0.6

    return jsonify({'valence': valence, 'arousal': arousal}), 200

@music.route('/api/music/vote/<music_id>', methods=['POST'])
def vote_music(music_id):
    data = request.json
    user_id = data.get('user_id')
    vote = data.get('vote')

    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({'code': 401, 'message': 'User not logged in'}), 401

    music = Music.query.filter_by(id=music_id).first()
    if not music:
        return jsonify({'code': 404, 'message': 'Music not found'}), 404

    new_behavior = UserBehavior(user_id=user.id, music_id=music.id, rate=float(vote))
    db.session.add(new_behavior)
    db.session.commit()

    return jsonify({'code': 200, 'message': 'Vote successfully.'}), 200

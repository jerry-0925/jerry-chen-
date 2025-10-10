from flask import Blueprint, request, jsonify
from .models import db, User, Task
import uuid

tasks = Blueprint('tasks', __name__)

@tasks.route('/api/suno/user_id/task_id', methods=['POST'])
def generate_music():
    data = request.json
    user_id = data.get('user_id')
    prompt = data.get('prompt')
    instrumental = data.get('instrumental', False)

    user = User.query.filter_by(id=user_id).first()
    if not user:
        return jsonify({'code': 401, 'message': 'User ID invalid'}), 401

    task_id = str(uuid.uuid4())
    new_task = Task(id=task_id, token_id=user_id, prompt=prompt, status='pending')
    db.session.add(new_task)
    db.session.commit()

    return jsonify({'task_id': task_id, 'message': 'Task submitted successfully.'}), 200

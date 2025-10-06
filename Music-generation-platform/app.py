from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from forms import RegistrationForm, LoginForm
from music_processing import make_volcano_request, analyze_music
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from flask_login import login_user, logout_user, login_required, LoginManager, current_user
from flask_migrate import Migrate
from flask_login import UserMixin
app = Flask(__name__)
app.config.from_object('config.Config')
TASK_FILE = "tasks.json"
# Initialize the database
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Load models
MODEL_PATH = ""
model = joblib.load(MODEL_PATH)
MODEL_PATH2 = ''
model2 = joblib.load(MODEL_PATH2)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Specify the login route to redirect unauthenticated users

# Define the User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Song(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    artist = db.Column(db.String(100), nullable=False)
    genre = db.Column(db.String(50), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    valence = db.Column(db.Float, nullable=True)
    arousal = db.Column(db.Float, nullable=True)

class UserActivity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    song_id = db.Column(db.Integer, db.ForeignKey('song.id'), nullable=False)
    action = db.Column(db.String(10), nullable=False)  # 'like' or 'dislike'

    user = db.relationship('User', backref=db.backref('activities', lazy=True))
    song = db.relationship('Song', backref=db.backref('activities', lazy=True))


# Initialize database tables
with app.app_context():
    db.create_all()

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return []

def save_json(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

@app.route('/')
@app.route('/home')
@login_required  # Protect this route with Flask-Login
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        # Check if user already exists
        existing_user = User.query.filter_by(email=form.email.data).first()
        if existing_user:
            flash('Email already exists', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created!', 'success')
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        # Verify user exists and password matches
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)  # Use Flask-Login to log in the user
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html', form=form)


@app.route('/dashboard')
@login_required  # Protect this route with Flask-Login
def dashboard():
    tasks = load_json(TASK_FILE)
    user_tasks = [task for task in tasks if task['username'] == current_user.username]
    return render_template('dashboard.html', tasks=user_tasks)


@app.route('/songs', methods=['GET', 'POST'])
@login_required  # Protect this route with Flask-Login
def songs():
    # Retrieve the user object
    user = User.query.filter_by(username=current_user.username).first()

    if not user:
        flash('User not found. Please log in again.', 'danger')
        return redirect(url_for('login'))

    # Generate recommendations using the user ID
    user_recommendations = generate_recommendations(user.id)

    # Get search query from form
    query = request.form.get('search', '').strip()
    if query:
        # Filter songs based on title, genre, or artist
        songs = Song.query.filter(
            (Song.title.ilike(f'%{query}%')) |
            (Song.artist.ilike(f'%{query}%')) |
            (Song.genre.ilike(f'%{query}%'))
        ).all()
    else:
        # Get all songs if no search query
        songs = Song.query.all()

    return render_template('songs.html', songs=songs, query=query, recommendations=user_recommendations)


@app.route('/play_song/<int:song_id>')
@login_required  # Protect this route with Flask-Login
def play_song(song_id):
    song = Song.query.get_or_404(song_id)
    return render_template('play_song.html', song=song)


@app.route('/api/songs', methods=['GET', 'POST'])
@login_required  # Protect this route with Flask-Login
def search_songs_api():
    user = User.query.filter_by(username=current_user.username).first()

    if request.method == 'POST':
        search_query = request.json.get('search', '').strip().lower()
        if search_query:
            songs = Song.query.filter(
                (Song.title.ilike(f"%{search_query}%")) |
                (Song.artist.ilike(f"%{search_query}%")) |
                (Song.genre.ilike(f"%{search_query}%"))
            ).all()
        else:
            songs = Song.query.all()  # Fetch all songs if no query
    else:
        songs = Song.query.all()  # Fetch all songs for GET

    songs_data = []
    for song in songs:
        # Check user's activity on this song
        activity = UserActivity.query.filter_by(user_id=user.id, song_id=song.id).first()
        liked = activity.action == 'like' if activity else False
        disliked = activity.action == 'dislike' if activity else False

        songs_data.append({
            'id': song.id,
            'title': song.title,
            'artist': song.artist,
            'genre': song.genre,
            'valence': song.valence,
            'arousal': song.arousal,
            'liked': liked,
            'disliked': disliked
        })

    return jsonify(songs_data)


@app.route('/like_dislike/<int:song_id>', methods=['POST'])
@login_required  # Protect this route with Flask-Login
def like_dislike(song_id):
    song = Song.query.get_or_404(song_id)
    action = request.json.get('action')  # 'like' or 'dislike'

    if action not in ['like', 'dislike']:
        return jsonify({"error": "Invalid action"}), 400

    existing_activity = UserActivity.query.filter_by(user_id=current_user.id, song_id=song.id).first()

    if existing_activity:
        if existing_activity.action == action:
            # Toggle off the same action
            db.session.delete(existing_activity)
        else:
            # Switch to the other action
            existing_activity.action = action
    else:
        # Create a new activity
        new_activity = UserActivity(user_id=current_user.id, song_id=song.id, action=action)
        db.session.add(new_activity)

    db.session.commit()
    return jsonify({"success": True, "message": f"Song {action}d successfully"})


@app.route('/logout')
@login_required  # Protect this route with Flask-Login
def logout():
    logout_user()  # Logout the user and clear the session
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))  # Redirect to login after logging out


@app.route('/generate_music', methods=['POST'])
def generate_music_route():
    # Extract data from the form
    prompt = request.form['prompt']
    genre = request.form.get('genre')  # Get selected genre
    mood = request.form.get('mood')  # Get selected mood
    timbre = request.form.get('timbre')  # Get selected timbre
    gender = request.form.get('gender')  # Get selected gender

    body = {
        "Prompt": prompt,
        "Genre": genre,
        "Mood": mood,
        "Gender": gender,
        "Timbre": timbre,
    }

    # Pass the inputs to the generate_music function
    result = make_volcano_request(
        method="POST",
        action="GenSongV4",
        query={},
	body=body
    )
    print(result)

    # Update the task record
    tasks = load_json(TASK_FILE)
    task = {
        "username": current_user.username,
        "prompt": prompt,
        "genre": genre,
        "mood": mood,
        "timbre": timbre,
        "gender": gender,
        "result": result,
        "file": None
    }
    tasks.append(task)
    save_json(TASK_FILE, tasks)

    flash('Music generated!', 'success')
    return redirect(url_for('dashboard'))


@app.route('/view_task/<int:task_id>')
@login_required  # Protect this route with Flask-Login
def view_task(task_id):
    tasks = load_json(TASK_FILE)
    user_tasks = [task for task in tasks if task['username'] == current_user.username]

    if task_id < 0 or task_id >= len(user_tasks):
        flash('Task not found.', 'danger')
        return redirect(url_for('dashboard'))

    task = user_tasks[task_id]
    return render_template('view_task.html', task=task)


@app.route('/upload_music', methods=['GET', 'POST'])
@login_required  # Protect this route with Flask-Login
def upload_music():
    if request.method == 'POST':
        file = request.files['file']
        if not os.path.exists('uploads'):
            os.mkdir("uploads")
        file_path = f"uploads/{file.filename}"
        file.save(file_path)
        result = analyze_music(model, model2, file_path, file.filename)

        tasks = load_json(TASK_FILE)
        task = {
            "username": current_user.username,
            "prompt": None,
            "result": result,
            "file": file.filename
        }
        tasks.append(task)
        save_json(TASK_FILE, tasks)
        flash('Music analyzed!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('upload.html')


def generate_recommendations(user_id):
    user_activities = UserActivity.query.filter_by(user_id=user_id).all()
    liked_songs = [activity.song for activity in user_activities if activity.action == 'like']
    disliked_songs = [activity.song for activity in user_activities if activity.action == 'dislike']

    liked_features = np.array([[song.valence, song.arousal] for song in liked_songs])
    disliked_features = np.array([[song.valence, song.arousal] for song in disliked_songs])

    all_songs = Song.query.all()
    all_song_features = np.array([[song.valence, song.arousal] for song in all_songs])

    liked_similarities = cosine_similarity(liked_features, all_song_features).mean(axis=0) if liked_features.size > 0 else np.zeros(len(all_songs))
    disliked_similarities = cosine_similarity(disliked_features, all_song_features).mean(axis=0) if disliked_features.size > 0 else np.zeros(len(all_songs))

    final_similarities = liked_similarities - disliked_similarities
    sorted_indices = np.argsort(final_similarities)[::-1]

    interacted_song_ids = {activity.song_id for activity in user_activities}
    recommendations = []
    for idx in sorted_indices:
        song = all_songs[idx]
        if song.id not in interacted_song_ids:
            recommendations.append(song)
        if len(recommendations) == 5:
            break

    return recommendations


@app.route('/view_recommendations')
@login_required  # Protect this route with Flask-Login
def view_recommendations():
    recommended_songs = generate_recommendations(current_user.id)
    return render_template('recommendations.html', recommended_songs=recommended_songs)


if __name__ == '__main__':
    app.secret_key = app.config['SECRET_KEY']
    app.run(debug=True)

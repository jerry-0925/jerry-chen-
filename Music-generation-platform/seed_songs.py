from app import db, Song, app

def seed_songs():
    # List of sample songs
    sample_songs = [
        {
            "title": "",
            "artist": "",
            "genre": ",
            "valence": 1.2943717772352419,
            "arousal": 4.496466787358037e-05,
            "file_path": ""
        },
        {
            "title": "",
            "artist": "",
            "genre": "",
            "valence": 1.0488033147380222,
            "arousal": 6.035301345286463e-05,
            "file_path": ""
        },

    ]

    # Add sample songs to the database
    for song in sample_songs:
        new_song = Song(
            title=song["title"],
            artist=song["artist"],
            genre=song["genre"],
            valence=song["valence"],
            arousal=song["arousal"],
            file_path=song["file_path"]
        )
        db.session.add(new_song)

    db.session.commit()
    print("Sample songs added to the database.")

if __name__ == "__main__":
    # Use Flask's application context
    with app.app_context():
        seed_songs()



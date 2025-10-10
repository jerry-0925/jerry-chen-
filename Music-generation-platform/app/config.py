class Config:
    SQLALCHEMY_DATABASE_URI = 'mysql://username:password@localhost/db_name'  # Update with your database URI
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'your-secret-key'  # Set a secret key for session management

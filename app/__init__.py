# This file is intentionally left empty to make the directory a Python package 

# Import the Flask app instance from app.py
import sys
import os
from flask import Flask

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'ppt', 'pptx', 'doc', 'docx', 'txt'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import routes after the app is created to avoid circular imports
from app import routes 
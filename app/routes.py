from flask import render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import os
from app import app
from app.services.video_service import extract_video_info, summarize_video
from app.services.document_service import process_document
from app.services.summary_service import generate_detailed_summary

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    video_url = request.form.get('video_url')
    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400
    
    try:
        # Extract video information
        video_info = extract_video_info(video_url)
        
        # Generate summary
        summary = summarize_video(video_url)
        
        return jsonify({
            "success": True,
            "video_info": video_info,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({"error": "No document provided"}), 400
    
    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "No document selected"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        try:
            # Process the document
            document_summary = process_document(file_path)
            
            return jsonify({
                "success": True,
                "document_summary": document_summary
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/generate_content', methods=['POST'])
def generate_content():
    data = request.get_json()
    
    if not data or 'sources' not in data or not data['sources']:
        return jsonify({"error": "No sources provided"}), 400
    
    try:
        # Generate detailed summary and content
        sources = data['sources']
        topic = data.get('topic', 'General Topic')
        
        detailed_summary = generate_detailed_summary(sources, topic)
        
        return jsonify({
            "success": True,
            "detailed_summary": detailed_summary
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500 
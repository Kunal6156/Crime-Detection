from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json
import requests

# Create Flask API application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global pipeline instance
pipeline = None

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "providers": list(pipeline.llm_manager.config.keys()) if pipeline else []
    })

@app.route('/providers', methods=['GET'])
def get_providers():
    """Get available LLM providers"""
    if not pipeline:
        return jsonify({"error": "Pipeline not initialized"}), 500
    
    return jsonify({
        "providers": [k for k in pipeline.llm_manager.config.keys() if k != "default_provider"],
        "default_provider": pipeline.llm_manager.default_provider
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint to upload media files for analysis"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    return jsonify({
        "status": "success",
        "file_path": file_path,
        "filename": filename
    })

@app.route('/analyze/image/path', methods=['POST'])
def analyze_image_path():
    """API endpoint to analyze image from a file path"""
    data = request.json
    if not data or 'image_path' not in data:
        return jsonify({"error": "image_path is required"}), 400
    
    image_path = data['image_path']
    provider = data.get('provider', pipeline.llm_manager.default_provider)
    async_mode = data.get('async', False)
    callback_url = data.get('callback_url')
    
    if async_mode:
        # Queue the job and return immediately
        queue_result = pipeline.queue_analysis('image_file', image_path, provider, callback_url)
        return jsonify(queue_result)
    else:
        # Process synchronously
        result = pipeline.analyze_image_path(image_path, provider)
        return jsonify(result)

@app.route('/analyze/image/url', methods=['POST'])
def analyze_image_url():
    """API endpoint to analyze image from a URL"""
    data = request.json
    if not data or 'image_url' not in data:
        return jsonify({"error": "image_url is required"}), 400
    
    image_url = data['image_url']
    provider = data.get('provider', pipeline.llm_manager.default_provider)
    async_mode = data.get('async', False)
    callback_url = data.get('callback_url')
    
    if async_mode:
        # Queue the job and return immediately
        queue_result = pipeline.queue_analysis('image_url', image_url, provider, callback_url)
        return jsonify(queue_result)
    else:
        # Process synchronously
        result = pipeline.analyze_image_url(image_url, provider)
        return jsonify(result)

@app.route('/analyze/audio', methods=['POST'])
def analyze_audio():
    """API endpoint to analyze audio from a file path"""
    data = request.json
    if not data or 'audio_path' not in data:
        return jsonify({"error": "audio_path is required"}), 400
    
    audio_path = data['audio_path']
    provider = data.get('provider', pipeline.llm_manager.default_provider)
    async_mode = data.get('async', False)
    callback_url = data.get('callback_url')
    
    if async_mode:
        # Queue the job and return immediately
        queue_result = pipeline.queue_analysis('audio_file', audio_path, provider, callback_url)
        return jsonify(queue_result)
    else:
        # Process synchronously
        result = pipeline.analyze_audio_path(audio_path, provider)
        return jsonify(result)

@app.route('/analyze/video', methods=['POST'])
def analyze_video():
    """API endpoint to analyze video from a file path"""
    data = request.json
    if not data or 'video_path' not in data:
        return jsonify({"error": "video_path is required"}), 400
    
    video_path = data['video_path']
    provider = data.get('provider', pipeline.llm_manager.default_provider)
    frame_interval = data.get('frame_interval', 5)
    async_mode = data.get('async', True)  # Default to async for video due to processing time
    callback_url = data.get('callback_url')
    
    if async_mode:
        # Queue the job and return immediately
        queue_result = pipeline.queue_analysis('video_file', video_path, provider, callback_url)
        return jsonify(queue_result)
    else:
        # Process synchronously
        result = pipeline.analyze_video_path(video_path, provider)
        return jsonify(result)

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """API endpoint to check status of an async job"""
    result = pipeline.get_job_result(job_id)
    return jsonify(result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

def run_api_server(llm_config, host='0.0.0.0', port=5000):
    """Run the API server"""
    global pipeline
    pipeline = CrimeDetectionPipeline(llm_config)
    
    logger.info(f"Starting Multimodal Crime Detection API server on {host}:{port}")
    app.run(host=host, port=port)
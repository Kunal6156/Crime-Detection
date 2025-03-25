import os
import sys
import argparse

# Add the project root and parent directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)

# Dynamically import based on project structure
try:
    from app.api import run_api_server
    from app.crime_detection import CrimeDetectionPipeline
except ImportError:
    try:
        from MINI_PROJECT.app.api import run_api_server
        from MINI_PROJECT.app.crime_detection import CrimeDetectionPipeline
    except ImportError:
        print("Error: Unable to import required modules. Check your project structure.")
        sys.exit(1)

# Import configuration
try:
    import json
    with open(os.path.join(project_root, 'config.json'), 'r') as config_file:
        llm_config = json.load(config_file)
except FileNotFoundError:
    print("Error: config.json not found. Please create it with LLM provider configurations.")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error: Invalid config.json format.")
    sys.exit(1)

def process_media(input_path):
    """Process media file using the Crime Detection Pipeline"""
    try:
        # Initialize the pipeline with LLM configuration
        pipeline = CrimeDetectionPipeline(llm_config)
        
        # Determine media type based on file extension
        file_extension = os.path.splitext(input_path)[1].lower()
        
        # Map extensions to appropriate analysis methods
        if file_extension in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            return pipeline.analyze_image_path(input_path)
        elif file_extension in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
            return pipeline.analyze_audio_path(input_path)
        elif file_extension in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            return pipeline.analyze_video_path(input_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    except Exception as e:
        print(f"Error processing media: {e}")
        raise

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Crime Detection Pipeline')
    parser.add_argument('--mode', choices=['web', 'process'], 
                        default='web', 
                        help='Run mode: web interface or direct media processing')
    parser.add_argument('--input', type=str, 
                        help='Path to input media file (for process mode)')
    
    # Parse arguments
    args = parser.parse_args()

    # Configure paths
    uploads_dir = os.path.join(project_root, 'uploads')
    temp_dir = os.path.join(project_root, 'temp')
    
    # Ensure directories exist
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Run based on mode
    if args.mode == 'web':
        # Start web interface
        try:
            run_api_server(llm_config, host='0.0.0.0', port=5000)
        except Exception as e:
            print(f"Error starting API server: {e}")
            sys.exit(1)
    
    elif args.mode == 'process':
        # Direct media processing
        if not args.input:
            print("Error: Input file path is required in process mode")
            sys.exit(1)
        
        # Validate input file
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist")
            sys.exit(1)
        
        # Process the media
        try:
            results = process_media(args.input)
            print("Detection Results:")
            print(json.dumps(results, indent=2))
        except Exception as e:
            print(f"Error processing media: {e}")
            sys.exit(1)

if __name__ == '__main__':
    main()

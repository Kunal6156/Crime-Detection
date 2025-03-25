# Initialize the crime detection application package

from . import api
from . import media_processor
from . import llm_manager
from . import crime_detection
from . import utils

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crime_detection.log"),
        logging.StreamHandler()
    ]
)

# Application-wide configuration
APP_CONFIG = {
    'UPLOAD_FOLDER': 'uploads',
    'TEMP_FOLDER': 'temp',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16 MB max upload size
    'ALLOWED_EXTENSIONS': {
        'image': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'],
        'audio': ['.mp3', '.wav', '.ogg', '.flac', '.m4a'],
        'video': ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    }
}

__all__ = [
    'api', 
    'media_processor', 
    'llm_manager', 
    'crime_detection', 
    'utils', 
    'APP_CONFIG'
]
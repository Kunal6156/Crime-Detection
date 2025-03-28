import os
import json
import re
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import base64
from datetime import datetime
from pathlib import Path

# For media processing
from PIL import Image
from io import BytesIO
import requests
import cv2
import numpy as np
import tempfile
from moviepy import VideoFileClip, AudioFileClip

import speech_recognition as sr
from pydub import AudioSegment

# Add logging configuration
logger = logging.getLogger(__name__)

# Define supported file extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
SUPPORTED_AUDIO_EXTENSIONS = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

# Define temporary folder (you might want to adjust this path)
TEMP_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)
class MediaProcessor:
    """Handles media loading, validation, and preprocessing"""
    
    @staticmethod
    def validate_file_path(file_path: str, supported_extensions: List[str]) -> bool:
        """Check if path exists and is a supported format"""
        path = Path(file_path)
        return path.exists() and path.is_file() and path.suffix.lower() in supported_extensions
    
    @staticmethod
    def validate_image_path(image_path: str) -> bool:
        """Check if path exists and is a supported image format"""
        return MediaProcessor.validate_file_path(image_path, SUPPORTED_IMAGE_EXTENSIONS)
    
    @staticmethod
    def validate_audio_path(audio_path: str) -> bool:
        """Check if path exists and is a supported audio format"""
        return MediaProcessor.validate_file_path(audio_path, SUPPORTED_AUDIO_EXTENSIONS)
    
    @staticmethod
    def validate_video_path(video_path: str) -> bool:
        """Check if path exists and is a supported video format"""
        return MediaProcessor.validate_file_path(video_path, SUPPORTED_VIDEO_EXTENSIONS)
    
    @staticmethod
    def load_image(image_path: str) -> Optional[Image.Image]:
        """Load an image from a file path with error handling"""
        try:
            return Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def load_image_from_url(image_url: str) -> Optional[Image.Image]:
        """Load an image from a URL with error handling"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as e:
            logger.error(f"Failed to load image from URL {image_url}: {str(e)}")
            return None
    
    @staticmethod
    def preprocess_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image if needed and ensure format compatibility"""
        # Resize large images to prevent API issues
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if in RGBA mode
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    
    @staticmethod
    def extract_audio_from_video(video_path: str) -> str:
        try:
            # Generate a unique filename for the extracted audio
            temp_audio_path = os.path.join(TEMP_FOLDER, f"audio_{int(time.time())}.wav")

             # Extract audio using MoviePy
            video = VideoFileClip(video_path)  # ✅ Use VideoFileClip directly
            audio = video.audio
            if audio is None:
                logger.error(f"No audio track found in video {video_path}")
                return None
        
            audio.write_audiofile(temp_audio_path, logger=None)

            return temp_audio_path
        except Exception as e:
            logger.error(f"Failed to extract audio from video {video_path}: {str(e)}")
            return None
   
    @staticmethod
    def extract_frames_from_video(video_path: str, interval: int = 5) -> List[str]:
        """Extract frames from video at specified interval (in seconds)"""
        try:
            frame_paths = []
            video = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            # Calculate frame indices to extract
            frame_indices = [int(i * fps) for i in range(0, int(duration), interval)]
            
            # If video is very short, take at least first, middle and last frame
            if len(frame_indices) < 3 and frame_count > 3:
                frame_indices = [0, frame_count // 2, frame_count - 1]
            
            # Extract frames
            for i, frame_idx in enumerate(frame_indices):
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = video.read()
                if success:
                    frame_path = os.path.join(TEMP_FOLDER, f"frame_{i}_{int(time.time())}_{os.path.basename(video_path)}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
            
            video.release()
            return frame_paths
        except Exception as e:
            logger.error(f"Failed to extract frames from video {video_path}: {str(e)}")
            return []
    
    @staticmethod
    def transcribe_audio(audio_path: str, max_duration_minutes: int = 5) -> str:
        """
        Enhanced audio transcription with large file handling
        
        Args:
            audio_path (str): Path to audio file
            max_duration_minutes (int): Maximum audio duration to process
        
        Returns:
            str: Transcribed text
        """
        try:
            # Validate audio file
            if not os.path.exists(audio_path):
                logger.error(f"Audio file does not exist: {audio_path}")
                return ""
            
            # Log file details for debugging
            logger.info(f"Transcribing audio file: {audio_path}")
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")

            # Check file size and duration
            audio = AudioSegment.from_file(audio_path)
            duration_minutes = len(audio) / (1000 * 60)  # Convert milliseconds to minutes
            
            logger.info(f"Audio duration: {duration_minutes:.2f} minutes")
            
            # Trim audio if too long
            if duration_minutes > max_duration_minutes:
                logger.warning(f"Audio exceeds {max_duration_minutes} minutes. Trimming.")
                audio = audio[:max_duration_minutes * 60 * 1000]  # Trim to max duration
                trim_path = audio_path + "_trimmed.wav"
                audio.export(trim_path, format="wav")
                audio_path = trim_path

            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file with noise adjustment
            with sr.AudioFile(audio_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio_data = recognizer.record(source)
            
            # Try multiple recognition methods with increased robustness
            recognition_methods = [
                ('Google', recognizer.recognize_google),
                ('Sphinx', recognizer.recognize_sphinx),
                ('Google Cloud', recognizer.recognize_google_cloud),
                ('Wit.ai', recognizer.recognize_wit)
            ]
            
            for method_name, recognition_func in recognition_methods:
                try:
                    logger.info(f"Attempting transcription with {method_name}")
                    text = recognition_func(audio_data)
                    logger.info(f"Successful transcription with {method_name}")
                    return text
                except Exception as e:
                    logger.warning(f"{method_name} transcription failed: {e}")
            
            logger.error("All transcription methods failed")
            return ""
        
        except Exception as e:
            logger.error(f"Comprehensive audio transcription error: {str(e)}")
            logger.exception("Detailed audio transcription error:")
            return ""
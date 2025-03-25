import os
import json
import re
import base64
import time
import queue
import threading
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image
from io import BytesIO

from .media_processor import MediaProcessor
from .llm_manager import LLMManager, LLMProvider
from .utils.cache import CrimeAnalysisCache

# Set up logging
logger = logging.getLogger("crime_detection")

class CrimeDetectionPipeline:
    """Main pipeline for crime detection across different media types"""
    
    # Define comprehensive crime categories
    CRIME_CATEGORIES = [
        "Theft/Burglary (including shoplifting, pickpocketing, home invasion)",
        "Vandalism/Property Damage (including graffiti, arson, destruction of property)",
        "Assault/Violence (including fighting, physical abuse, weapons)",
        "Drug-related (drug use, possession, dealing, manufacturing)",
        "Fraud/Scam (including identity theft, financial fraud, counterfeiting)",
        "Cybercrime (including hacking, phishing, online harassment)",
        "Traffic Violations (including DUI, speeding, reckless driving)",
        "Public Disorder (including public intoxication, disturbing the peace)",
        "Environmental Crime (including illegal dumping, pollution, wildlife crimes)",
        "Organized Crime (including gang activity, racketeering, trafficking)",
        "Trespassing (unauthorized entry, breaking and entering)",
        "Weapon-related (illegal possession, brandishing weapons)",
        "Terrorism/Extremism (including signs of planning, extremist activity)",
        "Harassment/Stalking (following, threatening behavior)",
        "Money Laundering (including suspicious financial transactions)",
        "Human Trafficking (exploitation, forced labor, sex trafficking)",
        "Child Exploitation (including child abuse, child labor)",
        "Hate Crime (crime motivated by prejudice)",
        "Murder/Homicide (including evidence of violent death)",
        "None detected"
    ]
    
    def __init__(self, llm_config: Dict[str, str]):
        """Initialize the pipeline with LLM configuration"""
        self.llm_manager = LLMManager(llm_config)
        
        # Initialize components
        self.media_processor = MediaProcessor()
        self.cache = CrimeAnalysisCache()
        
        # Create processing queue for async processing
        self.processing_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        # Dictionary to store callback functions for async jobs
        self.callbacks = {}
    
    def _get_content_hash(self, content) -> str:
        """Generate a simple hash for content for caching purposes"""
        if isinstance(content, Image.Image):
            # For images
            thumb = content.copy()
            thumb.thumbnail((100, 100))
            thumb = thumb.convert('RGB')
            
            # Convert to bytes and hash
            img_bytes = BytesIO()
            thumb.save(img_bytes, format='JPEG')
            return base64.b64encode(img_bytes.getvalue()).decode('utf-8')[:32]
        elif isinstance(content, str):
            # For text (transcriptions, etc.)
            return base64.b64encode(content.encode('utf-8')).decode('utf-8')[:32]
        else:
            # Default to timestamp if unknown type
            return f"content_{int(time.time())}"
    
    def _process_queue(self):
        """Background thread for processing the queue of media files"""
        while True:
            try:
                job = self.processing_queue.get()
                if job:
                    job_id, media_type, data, provider, callback_url = job
                    
                    if media_type == 'image_file':
                        result = self.analyze_image_path(data, provider)
                    elif media_type == 'image_url':
                        result = self.analyze_image_url(data, provider)
                    elif media_type == 'audio_file':
                        result = self.analyze_audio_path(data, provider)
                    elif media_type == 'video_file':
                        result = self.analyze_video_path(data, provider)
                    else:
                        result = {"error": "Unknown media type"}
                    
                    # Store result for callback
                    self.callbacks[job_id] = result
                    
                    # Call webhook if provided
                    if callback_url:
                        try:
                            requests.post(callback_url, json={"job_id": job_id, "result": result})
                        except Exception as e:
                            logger.error(f"Failed to call webhook {callback_url}: {str(e)}")
                    
                self.processing_queue.task_done()
            except Exception as e:
                logger.error(f"Error in processing thread: {str(e)}")
    
    def queue_analysis(self, media_type: str, data: str, provider: str, callback_url: Optional[str] = None) -> Dict:
        """Add an analysis job to the processing queue"""
        job_id = f"{media_type}_{int(time.time())}_{os.urandom(4).hex()}"
        self.processing_queue.put((job_id, media_type, data, provider, callback_url))
        return {
            "status": "queued", 
            "job_id": job_id, 
            "media_type": media_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_job_result(self, job_id: str) -> Dict:
        """Get the result of a completed job"""
        if job_id in self.callbacks:
            return self.callbacks[job_id]
        else:
            return {"status": "pending", "job_id": job_id}
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Extract and parse JSON from model response with multiple fallbacks"""
        # Try direct JSON parsing first
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON using regex pattern matching
        try:
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                return json.loads(json_match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Try to extract JSON with curly braces and everything in between
        try:
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                json_str = response_text[start:end+1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return structured error with raw text for debugging
        logger.warning(f"Failed to parse JSON from response: {response_text[:100]}...")
        return {
            "error": "Failed to parse model response",
            "raw_response": response_text[:500],  # Include truncated response for debugging
            "crime_detected": False,
            "crime_type": "None detected",
            "confidence": "low"
        }
    
    def _image_analysis_prompt(self) -> str:
        """Generate the prompt for image crime detection"""
        categories_str = "\n".join([f"- {cat}" for cat in self.CRIME_CATEGORIES])
        
        return f"""
        Analyze this image and determine if it shows visual evidence of a crime or criminal activity.
        
        Examine the image carefully for signs of:
        {categories_str}
        
        You must respond with ONLY a valid JSON object and nothing else. No explanations or additional text.
        
        The JSON must have this exact structure:
        {{
            "crime_detected": [true or false],
            "crime_type": "[primary category from the list above]",
            "sub_category": "[specific type of crime if identifiable]",
            "confidence": "[high, medium, or low]",
            "evidence_description": "[detailed description of visual evidence]",
            "potential_severity": "[high, medium, or low]",
            "key_visual_elements": ["list of specific visual elements that indicate crime"],
            "recommendations": "[brief recommendations for next steps]",
            "alternative_interpretations": "[possible non-criminal explanations for the visual content]"
        }}
        """
    
    def _audio_analysis_prompt(self, transcription: str) -> str:
        """Generate the prompt for audio crime detection"""
        categories_str = "\n".join([f"- {cat}" for cat in self.CRIME_CATEGORIES])
        
        return f"""
        Analyze this audio transcription and determine if it contains evidence of a crime or criminal activity:
        
        TRANSCRIPTION:
        "{transcription}"
        
        Examine the transcription carefully for signs of:
        {categories_str}
        
        You must respond with ONLY a valid JSON object and nothing else. No explanations or additional text.
        
        The JSON must have this exact structure:
        {{
            "crime_detected": [true or false],
            "crime_type": "[primary category from the list above]",
            "sub_category": "[specific type of crime if identifiable]",
            "confidence": "[high, medium, or low]",
            "evidence_description": "[detailed description of evidence from audio]",
            "key_phrases": ["list of specific phrases that indicate crime"],
            "potential_severity": "[high, medium, or low]",
            "context_analysis": "[analysis of the context of the conversation]",
            "recommendations": "[brief recommendations for next steps]",
            "alternative_interpretations": "[possible non-criminal explanations for the content]"
        }}
        """
    
    def _video_frame_analysis_prompt(self) -> str:
        """Generate the prompt for video frame analysis"""
        return self._image_analysis_prompt()
    
    def _video_summary_prompt(self, frame_results: List[Dict], audio_result: Dict) -> str:
        """Generate the prompt for summarizing video analysis results"""
        frame_results_str = json.dumps(frame_results, indent=2)
        audio_result_str = json.dumps(audio_result, indent=2)
        
        return f"""
        I need you to synthesize the results of analyzing multiple frames from a video and its audio track.
        
        FRAME ANALYSIS RESULTS:
        {frame_results_str}
        
        AUDIO ANALYSIS RESULTS:
        {audio_result_str}
        
        Based on these analyses, provide a comprehensive assessment of potential criminal activity in the video.
        
        You must respond with ONLY a valid JSON object and nothing else. No explanations or additional text.
        
        The JSON must have this exact structure:
        {{
            "crime_detected": [true or false],
            "crime_type": "[primary crime category detected]",
            "sub_categories": ["list of specific crime types detected"],
            "confidence": "[high, medium, or low]",
            "evidence_summary": "[comprehensive summary of evidence from both visual and audio]",
            "key_timestamps": ["approximate timestamps where evidence is strongest based on frame order"],
            "potential_severity": "[high, medium, or low]",
            "recommendations": "[detailed recommendations for next steps]",
            "reliability_assessment": "[assessment of the reliability of the detection]"
        }}
        """
    
    def analyze_image(self, image: Image.Image, provider: str = None) -> Dict:
        """Analyze an image for evidence of crime"""
        try:
            # Use default provider if none specified
            if not provider:
                provider = self.llm_manager.default_provider
            
            # Generate image hash for caching
            image_hash = self._get_content_hash(image)
            
            # Check cache first
            cached_result = self.cache.get(image_hash)
            if cached_result:
                logger.info("Returning cached image analysis result")
                return {**cached_result, "source": "cache"}
            
            # Generate prompt
            prompt = self._image_analysis_prompt()
            
            # Call LLM
            llm_response = self.llm_manager.call_llm(provider, prompt, image)
            
            if "error" in llm_response:
                logger.error(f"Error in image analysis: {llm_response['error']}")
                return llm_response
            
            # Parse the response
            result = self._parse_json_response(llm_response["text"])
            
            # Add metadata
            result["timestamp"] = datetime.now().isoformat()
            result["source"] = "api"
            result["provider"] = provider
            
            # Cache the result
            self.cache.set(image_hash, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Image analysis error: {str(e)}")
            return {"error": f"Image analysis error: {str(e)}"}
    
    def analyze_image_path(self, image_path: str, provider: str = None) -> Dict:
        """Analyze an image from a file path"""
        # Validate path
        if not self.media_processor.validate_image_path(image_path):
            return {"error": f"Invalid image path or unsupported format: {image_path}"}
        
        # Load and preprocess image
        image = self.media_processor.load_image(image_path)
        if not image:
            return {"error": f"Failed to load image from path: {image_path}"}
        
        image = self.media_processor.preprocess_image(image)
        
        # Run analysis pipeline
        result = self.analyze_image(image, provider)
        result["image_path"] = image_path
        
        return result
    
    def analyze_image_url(self, image_url: str, provider: str = None) -> Dict:
        """Analyze an image from a URL"""
        # Load and preprocess image
        image = self.media_processor.load_image_from_url(image_url)
        if not image:
            return {"error": f"Failed to load image from URL: {image_url}"}
        
        image = self.media_processor.preprocess_image(image)
        
        # Run analysis pipeline
        result = self.analyze_image(image, provider)
        result["image_url"] = image_url
        
        return result
    
    def analyze_audio(self, audio_path: str, provider: str = None) -> Dict:
        """Analyze audio for evidence of crime"""
        try:
            # Use default provider if none specified
            if not provider:
                provider = self.llm_manager.default_provider
            
            # Generate content hash for caching
            audio_hash = f"audio_{os.path.basename(audio_path)}_{os.path.getsize(audio_path)}"
            
            # Check cache first
            cached_result = self.cache.get(audio_hash)
            if cached_result:
                logger.info("Returning cached audio analysis result")
                return {**cached_result, "source": "cache"}
            
            # Transcribe audio
            transcription = self.media_processor.transcribe_audio(audio_path)
            if not transcription:
                return {"error": "Failed to transcribe audio or no speech detected"}
            
            # Generate prompt
            prompt = self._audio_analysis_prompt(transcription)
            
            # Call LLM (text-only for audio transcription)
            llm_response = self.llm_manager.call_llm(provider, prompt)
            
            if "error" in llm_response:
                logger.error(f"Error in audio analysis: {llm_response['error']}")
                return llm_response
            
            # Parse the response
            result = self._parse_json_response(llm_response["text"])
            
            # Add metadata and transcription
            result["timestamp"] = datetime.now().isoformat()
            result["source"] = "api"
            result["provider"] = provider
            result["transcription"] = transcription
            
            # Cache the result
            self.cache.set(audio_hash, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis error: {str(e)}")
            return {"error": f"Audio analysis error: {str(e)}"}
    
    def analyze_audio_path(self, audio_path: str, provider: str = None) -> Dict:
        """Analyze an audio file from a path"""
        # Validate path
        if not self.media_processor.validate_audio_path(audio_path):
            return {"error": f"Invalid audio path or unsupported format: {audio_path}"}
        
        # Run analysis pipeline
        result = self.analyze_audio(audio_path, provider)
        result["audio_path"] = audio_path
        
        return result
    
    def analyze_video(self, video_path: str, provider: str = None, frame_interval: int = 5) -> Dict:
        """Analyze video for evidence of crime by processing both frames and audio"""
        try:
            # Use default provider if none specified
            if not provider:
                provider = self.llm_manager.default_provider
            
            # Generate content hash for caching
            video_hash = f"video_{os.path.basename(video_path)}_{os.path.getsize(video_path)}"
            
            # Check cache first
            cached_result = self.cache.get(video_hash)
            if cached_result:
                logger.info("Returning cached video analysis result")
                return {**cached_result, "source": "cache"}
            
            # Process video: extract frames and audio
            logger.info(f"Extracting frames from video: {video_path}")
            frame_paths = self.media_processor.extract_frames_from_video(video_path, frame_interval)
            if not frame_paths:
                return {"error": "Failed to extract frames from video"}
            
            logger.info(f"Extracting audio from video: {video_path}")
            audio_path = self.media_processor.extract_audio_from_video(video_path)
            if not audio_path:
                return {"error": "Failed to extract audio from video"}
            
            # Analyze each frame
            logger.info(f"Analyzing {len(frame_paths)} frames from video")
            frame_results = []
            for i, frame_path in enumerate(frame_paths):
                logger.info(f"Analyzing frame {i+1}/{len(frame_paths)}")
                frame_result = self.analyze_image_path(frame_path, provider)
                # Add frame index and approx timestamp
                frame_result["frame_index"] = i
                frame_result["approx_timestamp"] = i * frame_interval
                frame_results.append(frame_result)
            
            # Analyze audio
            logger.info("Analyzing video audio track")
            audio_result = self.analyze_audio(audio_path, provider)
            
            # Synthesize results
            logger.info("Synthesizing video analysis results")
            prompt = self._video_summary_prompt(frame_results, audio_result)
            
            # Call LLM for synthesis (text-only)
            llm_response = self.llm_manager.call_llm(provider, prompt)
            
            if "error" in llm_response:
                logger.error(f"Error in video synthesis: {llm_response['error']}")
                # Return partial results even if synthesis fails
                return {
                    "error": f"Error in video synthesis: {llm_response['error']}",
                    "frame_results": frame_results,
                    "audio_result": audio_result
                }
            
            # Parse the response
            result = self._parse_json_response(llm_response["text"])
            
            # Add metadata and component results
            result["timestamp"] = datetime.now().isoformat()
            result["source"] = "api"
            result["provider"] = provider
            result["frame_count"] = len(frame_paths)
            result["frame_results"] = frame_results
            result["audio_result"] = audio_result
            
            # Cache the result
            self.cache.set(video_hash, result)
            
            # Clean up temporary files
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary frame: {str(e)}")
            
            try:
                os.remove(audio_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary audio: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Video analysis error: {str(e)}")
            return {"error": f"Video analysis error: {str(e)}"}
    
    def analyze_video_path(self, video_path: str, provider: str = None) -> Dict:
        """Analyze a video file from a path"""
        # Validate path
        if not self.media_processor.validate_video_path(video_path):
            return {"error": f"Invalid video path or unsupported format: {video_path}"}
        
        # Run analysis pipeline
        result = self.analyze_video(video_path, provider)
        result["video_path"] = video_path
        
        return result
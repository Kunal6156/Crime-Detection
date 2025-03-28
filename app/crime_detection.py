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

# Assuming these are imported from respective modules
from .media_processor import MediaProcessor
from .llm_manager import LLMManager, LLMProvider
from .utils.cache import CrimeAnalysisCache

# Set up logging
logger = logging.getLogger("crime_detection")

class CrimeDetectionPipeline:
    """
    Comprehensive Crime Detection Pipeline with Multi-Modal Analysis
    
    Key Features:
    - Cross-provider analysis
    - Multi-modal detection (image, video, audio)
    - Intelligent caching
    - Async processing
    - Detailed crime categorization
    """
    
    # Comprehensive crime categories
    CRIME_CATEGORIES = [
        "Theft/Burglary (including shoplifting, pickpocketing, home invasion)",
        "Vandalism/Property Damage (including graffiti, arson, destruction of property)",
        "Assault/Violence (including fighting, physical abuse, weapons)",
        "Traffic Violations (hit and run, reckless driving, DUI)",
        "Drug-related (drug use, possession, dealing, manufacturing)",
        "Fraud/Scam (including identity theft, financial fraud, counterfeiting)",
        "Cybercrime (including hacking, phishing, online harassment)",
        "Public Disorder (including public intoxication, disturbing the peace)",
        "Environmental Crime (including illegal dumping, pollution)",
        "Organized Crime (including gang activity, racketeering, trafficking)",
        "Trespassing (unauthorized entry, breaking and entering)",
        "Weapon-related (illegal possession, brandishing weapons)",
        "Terrorism/Extremism (including signs of planning, extremist activity)",
        "Harassment/Stalking (following, threatening behavior)",
        "Vehicle Accident (collision, crash, road incident)",
        "Workplace Accident (industrial, construction, machinery-related)",
        "Public Safety Incident (emergency, potential hazard)",
        "Medical Emergency (injury, potential life-threatening situation)",
        "Infrastructure Failure (structural collapse, critical system breakdown)",
        "Natural Disaster Aftermath (earthquake, flood, storm damage)",
        "Transportation Incident (train, boat, airplane accidents)",
        "Industrial Accident (chemical spill, equipment malfunction)",
        "Fire or Explosion Incident",
        "Pedestrian or Cyclist Incident",
        "Money Laundering",
        "Human Trafficking",
        "Child Exploitation",
        "Hate Crime",
        "Murder/Homicide",
        "None detected"
    ]
    
    def __init__(self, llm_config: Dict[str, str]):
        """
        Initialize the Crime Detection Pipeline
        
        Args:
            llm_config (Dict[str, str]): Configuration for Language Models
        """
        # LLM Management
        self.llm_manager = LLMManager(llm_config)
        
        # Core Components
        self.media_processor = MediaProcessor()
        self.cache = CrimeAnalysisCache()
        
        # Async Processing
        self.processing_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
        
        # Callbacks for async jobs
        self.callbacks = {}
        
        # Logging configuration
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure logging for detailed tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('crime_detection.log')
            ]
        )
    
    def _get_content_hash(self, content) -> str:
        """
        Generate a hash for content caching
        
        Args:
            content: Image or text content to hash
        
        Returns:
            str: Base64 encoded hash
        """
        if isinstance(content, Image.Image):
            # Thumbnail generation for consistent hashing
            thumb = content.copy()
            thumb.thumbnail((100, 100))
            thumb = thumb.convert('RGB')
            
            img_bytes = BytesIO()
            thumb.save(img_bytes, format='JPEG')
            return base64.b64encode(img_bytes.getvalue()).decode('utf-8')[:32]
        elif isinstance(content, str):
            return base64.b64encode(content.encode('utf-8')).decode('utf-8')[:32]
        else:
            return f"content_{int(time.time())}"
    
    def _process_queue(self):
        """
        Background thread for processing media analysis jobs
        Supports async processing with optional webhook callbacks
        """
        while True:
            try:
                job = self.processing_queue.get()
                if job:
                    job_id, media_type, data, provider, callback_url = job
                    
                    # Process based on media type
                    if media_type == 'image_file':
                        result = self.analyze_image_path(data, provider)
                    elif media_type == 'image_url':
                        result = self.analyze_image_url(data, provider)
                    elif media_type == 'audio_file':
                        result = self.analyze_audio_path(data, provider)
                    elif media_type == 'video_file':
                        result = self.analyze_video_path(data, provider)
                    else:
                        result = {"error": "Unsupported media type"}
                    
                    # Store result for callback
                    self.callbacks[job_id] = result
                    
                    # Webhook notification
                    if callback_url:
                        try:
                            requests.post(callback_url, json={
                                "job_id": job_id, 
                                "result": result
                            })
                        except Exception as e:
                            logger.error(f"Webhook call failed: {str(e)}")
                    
                self.processing_queue.task_done()
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}")
    
    def queue_analysis(self, media_type: str, data: str, provider: str = None, callback_url: Optional[str] = None) -> Dict:
        """
        Queue a media analysis job
        
        Args:
            media_type (str): Type of media to analyze
            data (str): Path or URL of media
            provider (str, optional): Specific LLM provider
            callback_url (str, optional): Webhook URL for async results
        
        Returns:
            Dict: Job queuing information
        """
        job_id = f"{media_type}_{int(time.time())}_{os.urandom(4).hex()}"
        self.processing_queue.put((job_id, media_type, data, provider, callback_url))
        return {
            "status": "queued", 
            "job_id": job_id, 
            "media_type": media_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_job_result(self, job_id: str) -> Dict:
        """
        Retrieve the result of an async job
        
        Args:
            job_id (str): Unique job identifier
        
        Returns:
            Dict: Job result or status
        """
        return self.callbacks.get(job_id, {"status": "pending", "job_id": job_id})
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """
        Robust JSON parsing with comprehensive error handling
        
        Args:
            response_text (str): Raw LLM response text
        
        Returns:
            Dict: Parsed and validated response
        """
        logger.info(f"Full LLM Response: {response_text}")
    
        try:
            # Advanced cleaning and parsing
            cleaned_response = response_text.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:-3]
            
            result = json.loads(cleaned_response)
            
            # Strict key validation
            required_keys = [
                'crime_detected', 
                'crime_type', 
                'confidence'
            ]
            for key in required_keys:
                if key not in result:
                    logger.warning(f"Missing key in parsed JSON: {key}")
                    
            return result
        except Exception as e:
            logger.error(f"JSON Parsing Error: {str(e)}")
            return {
                "crime_detected": False,
                "crime_type": "Unknown",
                "confidence": "low",
                "parsing_error": str(e),
                "raw_response": response_text
            }
    
    def _image_analysis_prompt(self) -> str:
        """
        Generate comprehensive image analysis prompt
        
        Returns:
            str: Detailed prompt for image crime detection
        """
        categories_str = "\n".join([f"- {cat}" for cat in self.CRIME_CATEGORIES])
        return f"""
        Analyze this image for comprehensive crime detection.

        Detailed Crime Detection Guidelines:
        {categories_str}

        Additional Analysis Considerations:
        - Examine for direct criminal acts
        - Look for aftermath evidence
        - Consider contextual and environmental clues
        - Assess potential severity and immediate risks

        Response Format (STRICT JSON):
        {{
        "crime_detected": [true or false],
        "crime_type": "[Primary crime category]",
        "sub_category": "[Specific crime type]",
        "confidence": "[high/medium/low]",
        "evidence_description": "[Detailed visual evidence description]",
        "potential_severity": "[high/medium/low]",
        "key_visual_elements": ["List of critical visual indicators"],
        "recommendations": "[Immediate action recommendations]",
        "alternative_interpretations": "[Possible non-criminal explanations]"
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
        """Enhanced video summary prompt with more comprehensive analysis"""
        frame_results_str = json.dumps(frame_results, indent=2)
        audio_result_str = json.dumps(audio_result, indent=2)
        
        return f"""
        COMPREHENSIVE VIDEO CRIME DETECTION SYNTHESIS
        
        FRAME ANALYSIS DETAILS:
        {frame_results_str}
        
        AUDIO ANALYSIS DETAILS:
        {audio_result_str}
        
        ADVANCED ANALYSIS REQUIREMENTS:
        1. Conduct a holistic multi-modal analysis of video content
        2. Identify potential criminal activities across visual and audio domains
        3. Assess cumulative evidence and contextual patterns
        4. Provide nuanced, comprehensive risk assessment

        Perform deep analysis considering:
        - Patterns across multiple frames
        - Correlation between visual and audio evidence
        - Contextual progression of potential criminal indicators
        - Subtle signs of preparatory or aftermath activities

        You must respond with a comprehensive JSON object:
        {{
            "comprehensive_crime_detection": {{
                "crime_detected": [true or false],
                "primary_crime_type": "[most relevant crime category]",
                "crime_sub_types": ["list of potential related criminal activities"],
                "confidence_level": "[high/medium/low]",
                "severity_assessment": "[high/medium/low]",
                
                "visual_evidence_summary": {{
                    "frame_count_analyzed": [number of frames],
                    "frames_with_suspicious_activity": [number of suspicious frames],
                    "most_critical_frames": ["list of frame indices with highest risk"],
                    "key_visual_indicators": ["list of visual crime indicators"]
                }},
                
                "audio_evidence_summary": {{
                    "speech_indicators": ["list of suspicious audio elements"],
                    "tone_and_context_analysis": "brief description of audio context"
                }},
                
                "comprehensive_risk_profile": {{
                    "immediate_risk": [true or false],
                    "recommended_actions": ["list of specific follow-up recommendations"],
                    "additional_investigation_needed": [true or false]
                }},
                
                "alternative_interpretations": ["list of non-criminal explanations"],
                "evidence_reliability": "[high/medium/low]"
            }}
        }}
        """
    
    # [Rest of the methods from the previous implementation remain the same]
    
    def analyze_video(self, video_path: str, provider: str = None, frame_interval: int = 1) -> Dict:
        """
        Enhanced video analysis with multi-provider cross-validation
        
        Args:
            video_path (str): Path to video file
            provider (str, optional): Specific LLM provider
            frame_interval (int, optional): Frame sampling interval
        
        Returns:
            Dict: Comprehensive video crime detection results
        """
        try:
            # Provider selection
            if not provider:
                providers = [p for p in self.llm_manager.config.keys() if p != "default_provider"]
            else:
                providers = [provider]
            
            logger.info(f"VIDEO ANALYSIS: {video_path}")
            logger.info(f"Analysis Providers: {providers}")
            
            # Video file validation
            if not os.path.exists(video_path):
                return {"error": "Video file not found"}
            
            # Frame extraction
            frame_paths = self.media_processor.extract_frames_from_video(video_path, frame_interval)
            if not frame_paths:
                return {"error": "Frame extraction failed"}
            
            # Audio extraction
            audio_path = self.media_processor.extract_audio_from_video(video_path)
            
            # Cross-provider frame analysis
            frame_results_by_provider = {}
            for current_provider in providers:
                frame_results = []
                for i, frame_path in enumerate(frame_paths):
                    try:
                        frame_result = self.analyze_image_path(frame_path, current_provider)
                        frame_result["frame_index"] = i
                        frame_result["approx_timestamp"] = i * frame_interval
                        frame_results.append(frame_result)
                    except Exception as frame_error:
                        logger.error(f"Frame {i} analysis failed: {str(frame_error)}")
                
                frame_results_by_provider[current_provider] = frame_results
            
            # Cross-validation analysis function
            def analyze_cross_validation(frame_results_dict):
                crime_confidence = {}
                for provider, results in frame_results_dict.items():
                    crime_frames = [r for r in results if r.get('crime_detected', [False])[0]]
                    crime_confidence[provider] = {
                        'total_frames': len(results),
                        'crime_frames': len(crime_frames),
                        'crime_percentage': (len(crime_frames) / len(results)) * 100
                    }
                return crime_confidence
            
            cross_validation_results = analyze_cross_validation(frame_results_by_provider)
            
            # Audio analysis
            audio_result = self.analyze_audio(audio_path) if audio_path else {"error": "No audio analysis"}
            
            # Comprehensive crime detection prompt
            comprehensive_prompt = f"""
            ADVANCED VIDEO CRIME DETECTION ANALYSIS
            
            CROSS-VALIDATION INSIGHTS:
            {json.dumps(cross_validation_results, indent=2)}
            
            COMPREHENSIVE DETECTION REQUIREMENTS:
            1. Ultra-sensitive multi-modal crime detection
            2. Analyze potential criminal activities
            3. Assess severity and immediate risks
            
            You must respond with an extremely detailed JSON object capturing the nuanced crime detection analysis.
            """
            
            # Multi-provider comprehensive analysis
            final_results = []
            for current_provider in providers:
                try:
                    llm_response = self.llm_manager.call_llm(current_provider, comprehensive_prompt)
                    final_result = self._parse_json_response(llm_response["text"])
                    final_result["provider"] = current_provider
                    final_results.append(final_result)
                except Exception as analysis_error:
                    logger.error(f"Analysis failed with {current_provider}: {str(analysis_error)}")
            
            # Result aggregation
            result = {
                "crime_detected": any(r.get('crime_detected', False) for r in final_results),
                "cross_validation_results": cross_validation_results,
                "provider_results": final_results,
                "frame_results": frame_results_by_provider,
                "audio_result": audio_result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cleanup temporary files
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except Exception as cleanup_error:
                    logger.warning(f"Frame cleanup failed: {str(cleanup_error)}")
            
            if audio_path:
                try:
                    os.remove(audio_path)
                except Exception as cleanup_error:
                    logger.warning(f"Audio cleanup failed: {str(cleanup_error)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Video analysis error: {str(e)}")
            return {
                "crime_detected": True,
                "error": str(e)
            }
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
    
    def analyze_video_path(self, video_path: str, provider: str = None) -> Dict:
        """Analyze a video file from a path"""
        # Validate path
        if not self.media_processor.validate_video_path(video_path):
            return {"error": f"Invalid video path or unsupported format: {video_path}"}
        
        # Run analysis pipeline
        result = self.analyze_video(video_path, provider)
        result["video_path"] = video_path
        
        return result
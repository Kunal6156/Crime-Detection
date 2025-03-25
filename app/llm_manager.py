import logging
import time
import base64
from typing import Dict, Optional
from io import BytesIO
from PIL import Image

import anthropic
import openai
from google.generativeai import GenerativeModel, configure as configure_gemini
from langchain_community.llms import Replicate
from langchain_community.llms import HuggingFaceEndpoint

from .utils.rate_limiter import RateLimiter

# Set up logging
logger = logging.getLogger("crime_detection")

# LLM Provider Enum
class LLMProvider:
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"

class LLMManager:
    """Manages different LLM providers and their API calls"""
    
    def __init__(self, config: Dict[str, str]):
        """Initialize with API keys for different providers"""
        self.config = config
        self.default_provider = config.get("default_provider", LLMProvider.GEMINI)
        
        # Initialize providers
        if LLMProvider.GEMINI in config:
            configure_gemini(api_key=config[LLMProvider.GEMINI])
        
        if LLMProvider.OPENAI in config:
            openai.api_key = config[LLMProvider.OPENAI]
        
        if LLMProvider.ANTHROPIC in config:
            self.anthropic_client = anthropic.Anthropic(api_key=config[LLMProvider.ANTHROPIC])
        
        # Initialize rate limiters for each provider
        self.rate_limiters = {
            provider: RateLimiter(5)  # 5 requests per minute default
            for provider in config.keys() 
            if provider not in ["default_provider"]
        }
    
    def _call_gemini(self, prompt: str, image: Optional[Image.Image] = None) -> Dict:
        """Call the Google Gemini API"""
        try:
            # Apply rate limiting
            self.rate_limiters[LLMProvider.GEMINI].wait_if_needed()
            
            # Initialize the model
            model = GenerativeModel('gemini-1.5-pro-latest')
            
            # Generate content
            if image:
                response = model.generate_content([prompt, image])
            else:
                response = model.generate_content(prompt)
            
            # Return the response text
            return {"text": response.text}
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return {"error": str(e)}
    
    def _call_openai(self, prompt: str, image: Optional[Image.Image] = None) -> Dict:
        """Call the OpenAI API (GPT-4 Vision for images or GPT-4 for text)"""
        try:
            # Apply rate limiting
            self.rate_limiters[LLMProvider.OPENAI].wait_if_needed()
            
            if image:
                # Convert PIL image to base64 string
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Call GPT-4 Vision API
                response = openai.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {"role": "system", "content": "You are a crime detection assistant."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ],
                    max_tokens=2000
                )
                
                return {"text": response.choices[0].message.content}
            else:
                # Call regular GPT-4 API for text
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a crime detection assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000
                )
                
                return {"text": response.choices[0].message.content}
                
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {"error": str(e)}
    
    def _call_anthropic(self, prompt: str, image: Optional[Image.Image] = None) -> Dict:
        """Call the Anthropic API (Claude 3)"""
        try:
            # Apply rate limiting
            self.rate_limiters[LLMProvider.ANTHROPIC].wait_if_needed()
            
            if image:
                # Convert PIL image to format for Claude
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                
                # Call Claude API with image
                message = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    system="You are a crime detection assistant.",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(buffer.getvalue()).decode("utf-8")}}
                        ]
                    }]
                )
            else:
                # Call Claude API for text only
                message = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    system="You are a crime detection assistant.",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
            
            return {"text": message.content[0].text}
                
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            return {"error": str(e)}
    
    def _call_replicate(self, prompt: str) -> Dict:
        """Call the Replicate API (for text-only analysis)"""
        try:
            # Apply rate limiting
            self.rate_limiters[LLMProvider.REPLICATE].wait_if_needed()
            
            # Initialize the model
            llm = Replicate(
                model="meta/llama-3-70b-instruct:2a82ebb0e5a4c9f2de2304e3a27839f3f6ec69707e301595bd9f9bcf159c0d68",
                input={"temperature": 0.1, "max_length": 2000}
            )
            
            # Generate content
            response = llm.predict(prompt)
            
            return {"text": response}
                
        except Exception as e:
            logger.error(f"Error calling Replicate API: {str(e)}")
            return {"error": str(e)}
    
    def _call_huggingface(self, prompt: str) -> Dict:
        """Call the HuggingFace Inference API (for text-only analysis)"""
        try:
            # Apply rate limiting
            self.rate_limiters[LLMProvider.HUGGINGFACE].wait_if_needed()
            
            # Initialize the model
            llm = HuggingFaceEndpoint(
                endpoint_url=f"https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
                huggingfacehub_api_token=self.config[LLMProvider.HUGGINGFACE],
                max_length=2000,
                temperature=0.1
            )
            
            # Generate content
            response = llm.predict(prompt)
            
            return {"text": response}
                
        except Exception as e:
            logger.error(f"Error calling HuggingFace API: {str(e)}")
            return {"error": str(e)}
    
    def call_llm(self, provider: str, prompt: str, image: Optional[Image.Image] = None) -> Dict:
        """Call the specified LLM provider or fallback to default"""
        if provider == LLMProvider.GEMINI and LLMProvider.GEMINI in self.config:
            return self._call_gemini(prompt, image)
        elif provider == LLMProvider.OPENAI and LLMProvider.OPENAI in self.config:
            return self._call_openai(prompt, image)
        elif provider == LLMProvider.ANTHROPIC and LLMProvider.ANTHROPIC in self.config:
            return self._call_anthropic(prompt, image)
        elif provider == LLMProvider.REPLICATE and LLMProvider.REPLICATE in self.config and image is None:
            return self._call_replicate(prompt)
        elif provider == LLMProvider.HUGGINGFACE and LLMProvider.HUGGINGFACE in self.config and image is None:
            return self._call_huggingface(prompt)
        else:
            # Fallback to default provider
            logger.warning(f"Provider {provider} not available, falling back to {self.default_provider}")
            return self.call_llm(self.default_provider, prompt, image)
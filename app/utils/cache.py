import time
import threading
import json
import base64
from typing import Dict, Optional

class CrimeAnalysisCache:
    """Simple cache for results to avoid redundant API calls"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, content_hash: str) -> Optional[Dict]:
        """Get cached result if available"""
        with self.lock:
            return self.cache.get(content_hash)
    
    def set(self, content_hash: str, result: Dict):
        """Store result in cache"""
        with self.lock:
            # Implement simple LRU by removing oldest entry if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[content_hash] = result
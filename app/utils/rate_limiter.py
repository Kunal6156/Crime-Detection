import time
import threading
import logging

class RateLimiter:
    """Simple rate limiter to prevent API abuse"""
    def __init__(self, rate_limit_per_minute):
        self.interval = 60 / rate_limit_per_minute
        self.last_request_time = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we need to throttle requests"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.interval:
                sleep_time = self.interval - time_since_last
                logging.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
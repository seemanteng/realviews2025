"""
Performance monitoring and optimization utilities
Provides caching, timing, and resource management capabilities
"""

import time
import functools
import logging
import psutil
import threading
from typing import Any, Callable, Dict, Optional, Union
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import weakref
import gc

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    request_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def add_request(self, duration: float, error: bool = False):
        """Add a new request timing"""
        self.request_count += 1
        self.total_time += duration
        self.avg_time = self.total_time / self.request_count
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        if error:
            self.error_count += 1

class LRUCache:
    """Thread-safe LRU Cache implementation with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Check if expired
                if current_time - self.timestamps[key] > self.ttl:
                    self._remove(key)
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Add item to cache"""
        with self.lock:
            current_time = time.time()
            
            if key in self.cache:
                # Update existing
                self.cache[key] = value
                self.timestamps[key] = current_time
                self.cache.move_to_end(key)
            else:
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove oldest
                    oldest_key = next(iter(self.cache))
                    self._remove(oldest_key)
                
                self.cache[key] = value
                self.timestamps[key] = current_time
    
    def _remove(self, key: str):
        """Remove item from cache"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self):
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }

class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = defaultdict(PerformanceMetrics)
        self.start_time = time.time()
        self.lock = threading.RLock()
    
    def record_request(self, endpoint: str, duration: float, error: bool = False):
        """Record a request timing"""
        with self.lock:
            self.metrics[endpoint].add_request(duration, error)
    
    def get_metrics(self, endpoint: Optional[str] = None) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics"""
        with self.lock:
            if endpoint:
                return {endpoint: self.metrics[endpoint]}
            return dict(self.metrics)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system resource metrics"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_available_mb': memory.available / (1024 * 1024),
                'uptime_seconds': time.time() - self.start_time
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {}
    
    def clear_metrics(self):
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()

# Global instances
_performance_cache = LRUCache(max_size=1000, ttl=3600)
_performance_monitor = PerformanceMonitor()

def get_cache() -> LRUCache:
    """Get the global cache instance"""
    return _performance_cache

def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor"""
    return _performance_monitor

def cache(ttl: float = 3600, max_size: int = 1000):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        func_cache = LRUCache(max_size=max_size, ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = func_cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            func_cache.put(key, result)
            
            return result
        
        # Expose cache methods
        wrapper.cache_clear = func_cache.clear
        wrapper.cache_info = func_cache.get_stats
        
        return wrapper
    
    return decorator

def timed(endpoint: Optional[str] = None):
    """Decorator for timing function execution"""
    def decorator(func: Callable) -> Callable:
        endpoint_name = endpoint or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error_occurred = False
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                raise
            finally:
                duration = time.time() - start_time
                _performance_monitor.record_request(endpoint_name, duration, error_occurred)
                
                # Log slow requests
                if duration > 1.0:  # Log requests slower than 1 second
                    logger.warning(f"Slow request: {endpoint_name} took {duration:.2f}s")
        
        return wrapper
    
    return decorator

@contextmanager
def resource_monitor(operation_name: str):
    """Context manager for monitoring resource usage"""
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024 * 1024)
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(f"Resource usage for {operation_name}: "
                   f"Duration: {duration:.2f}s, Memory delta: {memory_delta:.2f}MB")

class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self.lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                current_time = time.time()
                
                if self.state == 'open':
                    if current_time - self.last_failure_time >= self.timeout:
                        self.state = 'half-open'
                        logger.info(f"Circuit breaker for {func.__name__} is now half-open")
                    else:
                        raise Exception(f"Circuit breaker open for {func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    if self.state == 'half-open':
                        self.state = 'closed'
                        self.failure_count = 0
                        logger.info(f"Circuit breaker for {func.__name__} is now closed")
                    return result
                
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = current_time
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = 'open'
                        logger.error(f"Circuit breaker opened for {func.__name__} "
                                   f"after {self.failure_count} failures")
                    
                    raise
        
        return wrapper

def memory_cleanup():
    """Force garbage collection and memory cleanup"""
    collected = gc.collect()
    logger.debug(f"Garbage collector: collected {collected} objects")
    
    # Get memory usage
    memory = psutil.virtual_memory()
    logger.debug(f"Memory usage: {memory.percent}% ({memory.used / (1024**3):.2f}GB used)")

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = threading.RLock()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier"""
        with self.lock:
            current_time = time.time()
            window_start = current_time - 60  # 1 minute window
            
            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
            
            # Check if under limit
            if len(self.requests[identifier]) < self.requests_per_minute:
                self.requests[identifier].append(current_time)
                return True
            
            return False

# Global rate limiter
_rate_limiter = RateLimiter()

def rate_limit(requests_per_minute: int = 60, identifier_func: Optional[Callable] = None):
    """Decorator for rate limiting function calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = 'default'
            
            if not _rate_limiter.is_allowed(identifier):
                raise Exception(f"Rate limit exceeded for {func.__name__}")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report"""
    cache_stats = _performance_cache.get_stats()
    system_metrics = _performance_monitor.get_system_metrics()
    endpoint_metrics = _performance_monitor.get_metrics()
    
    # Convert metrics to serializable format
    serializable_metrics = {}
    for endpoint, metrics in endpoint_metrics.items():
        serializable_metrics[endpoint] = {
            'request_count': metrics.request_count,
            'avg_time': metrics.avg_time,
            'min_time': metrics.min_time,
            'max_time': metrics.max_time,
            'error_count': metrics.error_count,
            'error_rate': metrics.error_count / max(metrics.request_count, 1)
        }
    
    return {
        'cache_stats': cache_stats,
        'system_metrics': system_metrics,
        'endpoint_metrics': serializable_metrics,
        'timestamp': time.time()
    }
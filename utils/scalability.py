"""
Scalability utilities and distributed processing capabilities
Provides tools for handling large-scale operations and concurrent processing
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import queue
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass
from functools import partial
import pickle
import json
from pathlib import Path
import math

logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Container for task execution results"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None

@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing operations"""
    batch_size: int = 100
    max_workers: int = multiprocessing.cpu_count()
    max_concurrent_batches: int = 3
    chunk_size: int = 10
    timeout_seconds: int = 300
    retry_attempts: int = 3
    enable_progress_tracking: bool = True

class WorkerPool:
    """Thread pool for concurrent task processing"""
    
    def __init__(self, max_workers: int = None, thread_name_prefix: str = 'RealViews'):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix
        )
        self.active_tasks = {}
        self.completed_tasks = []
        self.lock = threading.Lock()
    
    def submit_task(self, task_id: str, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task for execution"""
        start_time = time.time()
        
        def wrapped_func():
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                task_result = TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    worker_id=threading.current_thread().name
                )
                
                with self.lock:
                    self.completed_tasks.append(task_result)
                
                return task_result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = str(e)
                
                task_result = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=error_msg,
                    execution_time=execution_time,
                    worker_id=threading.current_thread().name
                )
                
                with self.lock:
                    self.completed_tasks.append(task_result)
                
                logger.error(f"Task {task_id} failed: {error_msg}")
                return task_result
        
        future = self.executor.submit(wrapped_func)
        
        with self.lock:
            self.active_tasks[task_id] = future
        
        return future
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of specific task"""
        with self.lock:
            if task_id in self.active_tasks:
                future = self.active_tasks[task_id]
                if future.done():
                    return 'completed'
                elif future.running():
                    return 'running'
                else:
                    return 'pending'
            
            # Check completed tasks
            for task_result in self.completed_tasks:
                if task_result.task_id == task_id:
                    return 'completed'
            
            return None
    
    def get_completed_results(self) -> List[TaskResult]:
        """Get all completed task results"""
        with self.lock:
            return self.completed_tasks.copy()
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> List[TaskResult]:
        """Wait for all active tasks to complete"""
        with self.lock:
            futures = list(self.active_tasks.values())
        
        if futures:
            concurrent.futures.wait(futures, timeout=timeout)
        
        return self.get_completed_results()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the worker pool"""
        self.executor.shutdown(wait=wait)

class BatchProcessor:
    """High-performance batch processing engine"""
    
    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or BatchProcessingConfig()
        self.worker_pool = WorkerPool(max_workers=self.config.max_workers)
        self.progress_callback = None
        self.stats = {
            'total_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'start_time': None,
            'end_time': None
        }
    
    def process_batch(self, 
                     items: List[Any], 
                     process_func: Callable,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[TaskResult]:
        """Process items in batches with parallel execution"""
        
        self.stats['total_items'] = len(items)
        self.stats['processed_items'] = 0
        self.stats['failed_items'] = 0
        self.stats['start_time'] = time.time()
        self.progress_callback = progress_callback
        
        if len(items) == 0:
            return []
        
        # Create batches
        batches = self._create_batches(items, self.config.batch_size)
        
        # Submit batch processing tasks
        futures = []
        for i, batch in enumerate(batches):
            task_id = f"batch_{i}"
            future = self.worker_pool.submit_task(
                task_id,
                self._process_single_batch,
                batch,
                process_func,
                i
            )
            futures.append(future)
        
        # Wait for completion and collect results
        results = []
        completed_batches = 0
        
        for future in concurrent.futures.as_completed(futures, timeout=self.config.timeout_seconds):
            try:
                batch_result = future.result()
                if isinstance(batch_result.result, list):
                    results.extend(batch_result.result)
                else:
                    results.append(batch_result)
                
                completed_batches += 1
                
                if self.progress_callback:
                    progress = (completed_batches / len(batches)) * 100
                    self.progress_callback(completed_batches, len(batches))
                
            except concurrent.futures.TimeoutError:
                logger.error("Batch processing timed out")
                break
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
        
        self.stats['end_time'] = time.time()
        self.stats['processed_items'] = len([r for r in results if r.success])
        self.stats['failed_items'] = len([r for r in results if not r.success])
        
        return results
    
    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Split items into batches"""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    def _process_single_batch(self, batch: List[Any], process_func: Callable, batch_id: int) -> TaskResult:
        """Process a single batch of items"""
        try:
            results = []
            for i, item in enumerate(batch):
                try:
                    result = process_func(item)
                    results.append(TaskResult(
                        task_id=f"batch_{batch_id}_item_{i}",
                        success=True,
                        result=result
                    ))
                except Exception as e:
                    results.append(TaskResult(
                        task_id=f"batch_{batch_id}_item_{i}",
                        success=False,
                        error=str(e)
                    ))
            
            return TaskResult(
                task_id=f"batch_{batch_id}",
                success=True,
                result=results
            )
            
        except Exception as e:
            return TaskResult(
                task_id=f"batch_{batch_id}",
                success=False,
                error=str(e)
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        
        if stats['start_time'] and stats['end_time']:
            stats['total_time'] = stats['end_time'] - stats['start_time']
            if stats['total_time'] > 0:
                stats['items_per_second'] = stats['processed_items'] / stats['total_time']
        
        if stats['total_items'] > 0:
            stats['success_rate'] = stats['processed_items'] / stats['total_items']
            stats['failure_rate'] = stats['failed_items'] / stats['total_items']
        
        return stats

class DistributedCache:
    """Simple distributed caching mechanism"""
    
    def __init__(self, cache_dir: Path = None, max_size_mb: int = 100):
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_mb = max_size_mb
        self.lock = threading.Lock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache"""
        cache_file = self.cache_dir / f"{key}.cache"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    
                # Check if expired
                if data['expires'] > time.time():
                    return data['value']
                else:
                    cache_file.unlink()  # Remove expired cache
            
            return default
            
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return default
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set item in cache"""
        try:
            with self.lock:
                # Clean up if cache is getting too large
                self._cleanup_if_needed()
                
                cache_file = self.cache_dir / f"{key}.cache"
                data = {
                    'value': value,
                    'created': time.time(),
                    'expires': time.time() + ttl
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                    
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
    
    def _cleanup_if_needed(self):
        """Clean up cache if it exceeds size limit"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
            max_size_bytes = self.max_size_mb * 1024 * 1024
            
            if total_size > max_size_bytes:
                # Remove oldest files
                cache_files = list(self.cache_dir.glob("*.cache"))
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                
                # Remove oldest 25% of files
                files_to_remove = cache_files[:len(cache_files) // 4]
                for file in files_to_remove:
                    file.unlink()
                    
        except Exception as e:
            logger.warning(f"Cache cleanup error: {e}")

class LoadBalancer:
    """Simple load balancer for distributing work"""
    
    def __init__(self, workers: List[Callable]):
        self.workers = workers
        self.current_worker = 0
        self.worker_stats = {i: {'requests': 0, 'errors': 0} for i in range(len(workers))}
        self.lock = threading.Lock()
    
    def get_worker(self) -> Callable:
        """Get next worker using round-robin"""
        with self.lock:
            worker = self.workers[self.current_worker]
            self.current_worker = (self.current_worker + 1) % len(self.workers)
            return worker
    
    def execute_with_fallback(self, *args, **kwargs) -> Any:
        """Execute task with automatic fallback to other workers"""
        for attempt in range(len(self.workers)):
            worker_index = (self.current_worker + attempt) % len(self.workers)
            worker = self.workers[worker_index]
            
            try:
                with self.lock:
                    self.worker_stats[worker_index]['requests'] += 1
                
                result = worker(*args, **kwargs)
                return result
                
            except Exception as e:
                with self.lock:
                    self.worker_stats[worker_index]['errors'] += 1
                
                logger.warning(f"Worker {worker_index} failed: {e}")
                
                if attempt == len(self.workers) - 1:  # Last attempt
                    raise e
        
        raise Exception("All workers failed")
    
    def get_stats(self) -> Dict[int, Dict[str, int]]:
        """Get worker statistics"""
        with self.lock:
            return self.worker_stats.copy()

class StreamingProcessor:
    """Process data streams efficiently"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.buffer = []
        self.lock = threading.Lock()
    
    def process_stream(self, 
                      data_stream: Iterator[Any], 
                      process_func: Callable,
                      flush_callback: Optional[Callable[[List[Any]], None]] = None) -> Iterator[Any]:
        """Process streaming data with buffering"""
        
        for item in data_stream:
            with self.lock:
                self.buffer.append(item)
                
                if len(self.buffer) >= self.buffer_size:
                    # Process buffer
                    batch = self.buffer.copy()
                    self.buffer.clear()
                    
                    # Process outside of lock
                    processed_batch = self._process_batch_streaming(batch, process_func)
                    
                    if flush_callback:
                        flush_callback(processed_batch)
                    
                    yield from processed_batch
        
        # Process remaining items
        if self.buffer:
            with self.lock:
                batch = self.buffer.copy()
                self.buffer.clear()
            
            processed_batch = self._process_batch_streaming(batch, process_func)
            
            if flush_callback:
                flush_callback(processed_batch)
            
            yield from processed_batch
    
    def _process_batch_streaming(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """Process batch for streaming"""
        results = []
        for item in batch:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing streaming item: {e}")
                results.append(None)  # or some error indicator
        
        return results

class ResourceMonitor:
    """Monitor system resources and adjust processing accordingly"""
    
    def __init__(self, max_memory_percent: float = 80.0, max_cpu_percent: float = 90.0):
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self.monitoring = False
        self.should_throttle = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor system resources"""
        try:
            import psutil
            
            while self.monitoring:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Check if resources are under pressure
                memory_pressure = memory.percent > self.max_memory_percent
                cpu_pressure = cpu_percent > self.max_cpu_percent
                
                self.should_throttle = memory_pressure or cpu_pressure
                
                if self.should_throttle:
                    logger.warning(f"Resource pressure detected - Memory: {memory.percent}%, CPU: {cpu_percent}%")
                
                time.sleep(5)  # Check every 5 seconds
                
        except ImportError:
            logger.warning("psutil not available - resource monitoring disabled")
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
    
    def should_throttle_processing(self) -> bool:
        """Check if processing should be throttled"""
        return self.should_throttle
    
    def adaptive_batch_size(self, base_batch_size: int) -> int:
        """Get adaptive batch size based on resource usage"""
        if self.should_throttle:
            return max(1, base_batch_size // 4)  # Reduce to 25%
        return base_batch_size

# Global instances for easy access
_batch_processor = BatchProcessor()
_distributed_cache = DistributedCache()
_resource_monitor = ResourceMonitor()

def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance"""
    return _batch_processor

def get_distributed_cache() -> DistributedCache:
    """Get global distributed cache instance"""
    return _distributed_cache

def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance"""
    return _resource_monitor

def parallel_map(func: Callable, items: List[Any], max_workers: int = None) -> List[Any]:
    """Parallel map function with optimal worker count"""
    if not items:
        return []
    
    if len(items) == 1:
        return [func(items[0])]
    
    max_workers = max_workers or min(len(items), multiprocessing.cpu_count())
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))

def chunked_processing(items: List[Any], chunk_size: int = None) -> Iterator[List[Any]]:
    """Split items into optimal chunks for processing"""
    if chunk_size is None:
        # Auto-calculate optimal chunk size
        chunk_size = max(1, len(items) // (multiprocessing.cpu_count() * 2))
    
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def memory_efficient_processing(items: Iterator[Any], 
                              process_func: Callable,
                              batch_size: int = 100) -> Iterator[Any]:
    """Process items in a memory-efficient manner"""
    batch = []
    
    for item in items:
        batch.append(item)
        
        if len(batch) >= batch_size:
            # Process batch
            for processed_item in parallel_map(process_func, batch):
                yield processed_item
            
            batch = []  # Clear batch to free memory
    
    # Process remaining items
    if batch:
        for processed_item in parallel_map(process_func, batch):
            yield processed_item
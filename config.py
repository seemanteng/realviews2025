"""
Configuration management for RealViews application
Centralizes all configuration settings with proper defaults and validation
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

@dataclass
class ModelConfig:
    """Machine Learning model configuration"""
    # Model performance settings
    confidence_threshold: float = 0.7
    quality_threshold: float = 0.5
    ensemble_voting: str = 'soft'  # 'hard' or 'soft'
    
    # Feature extraction settings
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 2)
    min_text_length: int = 3
    max_text_length: int = 5000
    
    # Model paths (relative to project root)
    model_base_path: Path = Path("models/saved_models")
    latest_model_version: Optional[str] = None
    
    # Training settings
    random_state: int = 42
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1

@dataclass
class PerformanceConfig:
    """Performance and caching configuration"""
    # Caching settings
    enable_model_caching: bool = True
    enable_prediction_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size: int = 1000
    
    # Batch processing
    batch_size: int = 100
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    
    # Memory management
    max_memory_usage_mb: int = 512
    garbage_collect_threshold: int = 100
    
    # Performance monitoring
    enable_metrics: bool = True
    log_slow_requests: bool = True
    slow_request_threshold_ms: int = 1000

@dataclass
class SecurityConfig:
    """Security and privacy configuration"""
    # Input validation
    max_input_length: int = 5000
    allowed_file_types: List[str] = field(default_factory=lambda: ['csv', 'txt'])
    max_file_size_mb: int = 10
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    
    # Data privacy
    log_user_inputs: bool = False
    anonymize_logs: bool = True
    data_retention_days: int = 30
    
    # API security
    require_api_key: bool = False
    api_key_header: str = "X-API-Key"
    
    # Sanitization
    sanitize_inputs: bool = True
    block_malicious_patterns: bool = True

@dataclass
class DatabaseConfig:
    """Database and storage configuration"""
    # File storage
    data_directory: Path = Path("data")
    processed_data_dir: Path = Path("data/processed")
    raw_data_dir: Path = Path("data/raw")
    models_dir: Path = Path("models/saved_models")
    
    # Backup settings
    enable_backups: bool = True
    backup_interval_hours: int = 24
    max_backup_files: int = 7
    
    # Data validation
    validate_data_integrity: bool = True
    check_data_corruption: bool = True

@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    # Log levels
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Log files
    log_directory: Path = Path("logs")
    log_file_name: str = "realviews.log"
    max_log_file_size_mb: int = 10
    backup_count: int = 5
    
    # Monitoring
    enable_performance_logging: bool = True
    enable_error_tracking: bool = True
    log_predictions: bool = False  # Privacy consideration

@dataclass
class TranslationConfig:
    """Translation service configuration"""
    # Translation settings
    default_target_language: str = "en"
    auto_detect_language: bool = True
    translation_timeout_seconds: int = 10
    
    # Supported languages
    supported_languages: List[str] = field(default_factory=lambda: [
        'en', 'zh', 'zh-cn', 'es', 'fr', 'de', 'ja', 'ko'
    ])
    
    # Quality settings
    min_translation_confidence: float = 0.8
    fallback_to_original: bool = True

@dataclass
class AppConfig:
    """Main application configuration"""
    # Application settings
    app_name: str = "RealViews"
    version: str = "1.0.0"
    debug_mode: bool = False
    
    # Streamlit settings
    page_title: str = "RealViews - ML Review Filter"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    
    def __post_init__(self):
        """Initialize configuration after object creation"""
        self._create_directories()
        self._setup_logging()
        self._load_environment_overrides()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.database.data_directory,
            self.database.processed_data_dir,
            self.database.raw_data_dir,
            self.database.models_dir,
            self.logging.log_directory
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=getattr(logging, self.logging.log_level),
            format=self.logging.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.logging.log_directory / self.logging.log_file_name
                )
            ]
        )
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        # Model settings
        if os.getenv('CONFIDENCE_THRESHOLD'):
            self.model.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))
        
        if os.getenv('BATCH_SIZE'):
            self.performance.batch_size = int(os.getenv('BATCH_SIZE'))
        
        if os.getenv('DEBUG_MODE'):
            self.debug_mode = os.getenv('DEBUG_MODE').lower() == 'true'
        
        if os.getenv('LOG_LEVEL'):
            self.logging.log_level = os.getenv('LOG_LEVEL')
        
        # Security settings
        if os.getenv('MAX_INPUT_LENGTH'):
            self.security.max_input_length = int(os.getenv('MAX_INPUT_LENGTH'))
        
        if os.getenv('ENABLE_RATE_LIMITING'):
            self.security.enable_rate_limiting = os.getenv('ENABLE_RATE_LIMITING').lower() == 'true'
    
    def validate_config(self) -> List[str]:
        """Validate configuration settings and return list of errors"""
        errors = []
        
        # Validate model config
        if not 0 <= self.model.confidence_threshold <= 1:
            errors.append("Model confidence_threshold must be between 0 and 1")
        
        if not 0 <= self.model.quality_threshold <= 1:
            errors.append("Model quality_threshold must be between 0 and 1")
        
        if self.model.tfidf_max_features < 100:
            errors.append("TF-IDF max_features should be at least 100")
        
        # Validate performance config
        if self.performance.batch_size < 1:
            errors.append("Batch size must be positive")
        
        if self.performance.max_concurrent_requests < 1:
            errors.append("Max concurrent requests must be positive")
        
        # Validate security config
        if self.security.max_input_length < 10:
            errors.append("Max input length should be at least 10 characters")
        
        if self.security.requests_per_minute < 1:
            errors.append("Rate limit must allow at least 1 request per minute")
        
        return errors
    
    def get_model_path(self, model_version: Optional[str] = None) -> Path:
        """Get path to specific model version or latest"""
        if model_version:
            return self.database.models_dir / model_version
        
        # Find latest model version
        if self.database.models_dir.exists():
            model_dirs = [d for d in self.database.models_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('v_')]
            if model_dirs:
                latest = max(model_dirs, key=lambda x: x.name)
                return latest
        
        # Fallback
        return self.database.models_dir / "latest"

# Global configuration instance
config = AppConfig()

def get_config() -> AppConfig:
    """Get the global configuration instance"""
    return config

def reload_config():
    """Reload configuration (useful for testing)"""
    global config
    config = AppConfig()
    return config
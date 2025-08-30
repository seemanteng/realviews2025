"""
Security utilities and input validation
Provides comprehensive security measures for the RealViews application
"""

import re
import html
import logging
import hashlib
import secrets
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import mimetypes
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    errors: List[str]
    sanitized_input: Optional[str] = None
    risk_level: str = "low"  # low, medium, high
    
    def add_error(self, error: str):
        """Add validation error"""
        self.is_valid = False
        self.errors.append(error)

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self):
        # Malicious patterns to block
        self.malicious_patterns = [
            # Script injection
            r'<\s*script[^>]*>.*?</\s*script\s*>',
            r'javascript\s*:',
            r'vbscript\s*:',
            r'on\w+\s*=',
            
            # SQL injection patterns
            r'(\b(union|select|insert|update|delete|drop|create|alter)\b)',
            r'(\-\-|\#|\/\*|\*\/)',
            r'(\b(or|and)\s+\d+\s*=\s*\d+)',
            
            # Command injection
            r'(\||&|;|\$\(|\`)',
            r'(\.\./|\.\.\\)',
            
            # XSS patterns
            r'(<\s*img[^>]*src\s*=)',
            r'(<\s*iframe[^>]*>)',
            r'(<\s*object[^>]*>)',
            r'(<\s*embed[^>]*>)',
            
            # Path traversal
            r'(\.\.\/|\.\.\\)',
            r'(\/etc\/|\/proc\/|\/sys\/)',
            
            # Excessive repetition (potential DoS)
            r'(.)\1{100,}',  # Same character repeated 100+ times
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                for pattern in self.malicious_patterns]
        
        # Safe file extensions
        self.safe_extensions = {'.txt', '.csv', '.json', '.yml', '.yaml'}
        
        # Maximum lengths
        self.max_lengths = {
            'review_text': 5000,
            'product_info': 500,
            'filename': 255,
            'general': 1000
        }
    
    def validate_text_input(self, text: str, input_type: str = 'general') -> ValidationResult:
        """Validate and sanitize text input"""
        result = ValidationResult(is_valid=True, errors=[])
        
        if not isinstance(text, str):
            result.add_error("Input must be a string")
            return result
        
        # Length validation
        max_length = self.max_lengths.get(input_type, self.max_lengths['general'])
        if len(text) > max_length:
            result.add_error(f"Input exceeds maximum length of {max_length} characters")
            result.risk_level = "medium"
        
        if len(text.strip()) == 0:
            result.add_error("Input cannot be empty")
        
        # Check for malicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                result.add_error("Input contains potentially malicious content")
                result.risk_level = "high"
                logger.warning(f"Malicious pattern detected in input: {pattern.pattern}")
        
        # Additional checks for suspicious content
        suspicion_score = self._calculate_suspicion_score(text)
        if suspicion_score > 0.7:
            result.add_error("Input appears to be suspicious")
            result.risk_level = "medium"
        
        # Sanitize input if valid
        if result.is_valid:
            result.sanitized_input = self._sanitize_text(text)
        
        return result
    
    def validate_file(self, file_path: Union[str, Path], file_content: Optional[bytes] = None) -> ValidationResult:
        """Validate uploaded file"""
        result = ValidationResult(is_valid=True, errors=[])
        
        path_obj = Path(file_path)
        
        # Validate filename
        if '..' in str(path_obj):
            result.add_error("Invalid file path")
            result.risk_level = "high"
        
        # Validate extension
        if path_obj.suffix.lower() not in self.safe_extensions:
            result.add_error(f"File type {path_obj.suffix} not allowed")
            result.risk_level = "medium"
        
        # Validate file size if content provided
        if file_content:
            max_size = 10 * 1024 * 1024  # 10MB
            if len(file_content) > max_size:
                result.add_error(f"File size exceeds {max_size // (1024*1024)}MB limit")
                result.risk_level = "medium"
            
            # Basic content validation
            try:
                # Try to decode as text for CSV/TXT files
                if path_obj.suffix.lower() in {'.txt', '.csv'}:
                    content_str = file_content.decode('utf-8')
                    text_validation = self.validate_text_input(content_str, 'general')
                    if not text_validation.is_valid:
                        result.errors.extend(text_validation.errors)
                        result.risk_level = max(result.risk_level, text_validation.risk_level, key=self._risk_level_weight)
            except UnicodeDecodeError:
                result.add_error("File contains invalid characters")
                result.risk_level = "medium"
        
        return result
    
    def validate_url(self, url: str) -> ValidationResult:
        """Validate URL input"""
        result = ValidationResult(is_valid=True, errors=[])
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in {'http', 'https'}:
                result.add_error("Only HTTP and HTTPS URLs are allowed")
                result.risk_level = "medium"
            
            # Check for suspicious domains
            suspicious_domains = {'localhost', '127.0.0.1', '0.0.0.0', '::1'}
            if parsed.hostname in suspicious_domains:
                result.add_error("Local URLs are not allowed")
                result.risk_level = "high"
            
            # Check URL length
            if len(url) > 2048:
                result.add_error("URL is too long")
                result.risk_level = "medium"
        
        except Exception:
            result.add_error("Invalid URL format")
            result.risk_level = "medium"
        
        return result
    
    def _calculate_suspicion_score(self, text: str) -> float:
        """Calculate suspicion score for text content"""
        score = 0.0
        
        # High ratio of special characters
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0:
            special_ratio = special_chars / len(text)
            if special_ratio > 0.3:
                score += 0.3
        
        # Excessive capitalization
        if len(text) > 0:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.5:
                score += 0.2
        
        # Repeated patterns
        if re.search(r'(.{3,})\1{3,}', text):
            score += 0.4
        
        # Too many URLs
        url_count = len(re.findall(r'https?://', text))
        if url_count > 2:
            score += 0.3
        
        return min(score, 1.0)
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text input"""
        # HTML escape
        sanitized = html.escape(text)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def _risk_level_weight(self, risk_level: str) -> int:
        """Get numeric weight for risk level"""
        weights = {'low': 1, 'medium': 2, 'high': 3}
        return weights.get(risk_level, 1)

class RateLimitManager:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.request_history = {}
        self.blocked_ips = set()
        self.block_duration = 3600  # 1 hour
    
    def is_allowed(self, identifier: str, endpoint: str = 'default') -> Tuple[bool, str]:
        """Check if request is allowed"""
        current_time = time.time()
        key = f"{identifier}:{endpoint}"
        
        # Check if blocked
        if identifier in self.blocked_ips:
            return False, "IP address temporarily blocked"
        
        # Initialize request history
        if key not in self.request_history:
            self.request_history[key] = []
        
        # Clean old requests (sliding window)
        window_start = current_time - 60  # 1 minute window
        self.request_history[key] = [
            req_time for req_time in self.request_history[key]
            if req_time > window_start
        ]
        
        # Check rate limits based on endpoint
        limits = self._get_rate_limits(endpoint)
        
        if len(self.request_history[key]) >= limits['requests_per_minute']:
            # Block if excessive requests
            if len(self.request_history[key]) > limits['requests_per_minute'] * 2:
                self.blocked_ips.add(identifier)
                logger.warning(f"Blocked IP {identifier} for excessive requests")
            
            return False, f"Rate limit exceeded: {limits['requests_per_minute']} requests per minute"
        
        # Allow request
        self.request_history[key].append(current_time)
        return True, "Request allowed"
    
    def _get_rate_limits(self, endpoint: str) -> Dict[str, int]:
        """Get rate limits for specific endpoint"""
        limits = {
            'default': {'requests_per_minute': 60},
            'analyze': {'requests_per_minute': 30},
            'batch_process': {'requests_per_minute': 10},
            'upload': {'requests_per_minute': 5},
        }
        
        return limits.get(endpoint, limits['default'])

class SecureSession:
    """Secure session management"""
    
    def __init__(self, session_timeout: int = 3600):
        self.session_timeout = session_timeout
        self.sessions = {}
    
    def create_session(self, user_id: str) -> str:
        """Create new secure session"""
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'csrf_token': secrets.token_urlsafe(16)
        }
        
        self.sessions[session_id] = session_data
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate and refresh session"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        # Check if expired
        if current_time - session['last_accessed'] > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        # Update last accessed
        session['last_accessed'] = current_time
        return session
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        self.sessions.pop(session_id, None)

class DataAnonymizer:
    """Data anonymization utilities"""
    
    @staticmethod
    def hash_identifier(identifier: str, salt: str = "realviews_salt") -> str:
        """Create one-way hash of identifier"""
        combined = f"{identifier}{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    @staticmethod
    def redact_personal_info(text: str) -> str:
        """Remove potential personal information from text"""
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers (basic patterns)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # URLs
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        
        # Potential addresses (simplified)
        text = re.sub(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b', '[ADDRESS]', text)
        
        return text

# Global instances
_validator = InputValidator()
_rate_limiter = RateLimitManager()

def validate_input(text: str, input_type: str = 'general') -> ValidationResult:
    """Validate text input using global validator"""
    return _validator.validate_text_input(text, input_type)

def validate_file_upload(file_path: Union[str, Path], content: Optional[bytes] = None) -> ValidationResult:
    """Validate file upload using global validator"""
    return _validator.validate_file(file_path, content)

def check_rate_limit(identifier: str, endpoint: str = 'default') -> Tuple[bool, str]:
    """Check rate limit using global rate limiter"""
    return _rate_limiter.is_allowed(identifier, endpoint)

def secure_hash(data: str) -> str:
    """Create secure hash of data"""
    return hashlib.sha256(data.encode()).hexdigest()

def generate_api_key() -> str:
    """Generate secure API key"""
    return secrets.token_urlsafe(32)

def constant_time_compare(a: str, b: str) -> bool:
    """Constant time string comparison to prevent timing attacks"""
    return secrets.compare_digest(a, b)

class SecurityMiddleware:
    """Security middleware for request processing"""
    
    def __init__(self):
        self.validator = InputValidator()
        self.rate_limiter = RateLimitManager()
    
    def process_request(self, request_data: Dict[str, Any], client_ip: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Process and validate request"""
        # Rate limiting
        allowed, message = self.rate_limiter.is_allowed(client_ip, request_data.get('endpoint', 'default'))
        if not allowed:
            return False, message, {}
        
        # Input validation
        sanitized_data = {}
        
        for key, value in request_data.items():
            if isinstance(value, str):
                validation = self.validator.validate_text_input(value, key)
                if not validation.is_valid:
                    return False, f"Invalid input for {key}: {', '.join(validation.errors)}", {}
                sanitized_data[key] = validation.sanitized_input
            else:
                sanitized_data[key] = value
        
        return True, "Request validated", sanitized_data

# Create global middleware instance
security_middleware = SecurityMiddleware()

def get_client_ip(request_headers: Dict[str, str]) -> str:
    """Extract client IP from request headers"""
    # Check for forwarded headers (when behind proxy)
    forwarded_headers = [
        'X-Forwarded-For',
        'X-Real-IP',
        'X-Client-IP',
        'CF-Connecting-IP'
    ]
    
    for header in forwarded_headers:
        if header in request_headers:
            # Take first IP if comma-separated
            ip = request_headers[header].split(',')[0].strip()
            if ip:
                return ip
    
    # Fallback to default
    return 'unknown'
"""
Test suite for security utilities
Ensures proper validation and security measures
"""

import unittest
import tempfile
from pathlib import Path
from utils.security import (
    InputValidator,
    ValidationResult,
    RateLimitManager,
    SecureSession,
    DataAnonymizer,
    validate_input,
    check_rate_limit,
    secure_hash
)


class TestInputValidator(unittest.TestCase):
    """Test input validation functionality"""
    
    def setUp(self):
        self.validator = InputValidator()
    
    def test_valid_text_input(self):
        """Test validation of valid text input"""
        result = self.validator.validate_text_input("This is a valid review text.")
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.risk_level, "low")
    
    def test_empty_input(self):
        """Test validation of empty input"""
        result = self.validator.validate_text_input("")
        self.assertFalse(result.is_valid)
        self.assertIn("Input cannot be empty", result.errors)
    
    def test_malicious_script_injection(self):
        """Test detection of script injection attempts"""
        malicious_input = "<script>alert('xss')</script>"
        result = self.validator.validate_text_input(malicious_input)
        self.assertFalse(result.is_valid)
        self.assertIn("Input contains potentially malicious content", result.errors)
        self.assertEqual(result.risk_level, "high")
    
    def test_sql_injection_detection(self):
        """Test detection of SQL injection patterns"""
        malicious_input = "'; DROP TABLE users; --"
        result = self.validator.validate_text_input(malicious_input)
        self.assertFalse(result.is_valid)
        self.assertEqual(result.risk_level, "high")
    
    def test_length_validation(self):
        """Test input length validation"""
        long_input = "a" * 6000  # Exceeds default max length
        result = self.validator.validate_text_input(long_input)
        self.assertFalse(result.is_valid)
        self.assertIn("Input exceeds maximum length", result.errors)
    
    def test_file_validation_safe_extension(self):
        """Test file validation with safe extension"""
        result = self.validator.validate_file("test.csv")
        self.assertTrue(result.is_valid)
    
    def test_file_validation_unsafe_extension(self):
        """Test file validation with unsafe extension"""
        result = self.validator.validate_file("malicious.exe")
        self.assertFalse(result.is_valid)
        self.assertIn("File type .exe not allowed", result.errors)
    
    def test_path_traversal_detection(self):
        """Test detection of path traversal attempts"""
        result = self.validator.validate_file("../../../etc/passwd")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.risk_level, "high")
    
    def test_url_validation_valid(self):
        """Test validation of valid URLs"""
        result = self.validator.validate_url("https://example.com/path")
        self.assertTrue(result.is_valid)
    
    def test_url_validation_invalid_scheme(self):
        """Test validation of URLs with invalid schemes"""
        result = self.validator.validate_url("ftp://example.com")
        self.assertFalse(result.is_valid)
        self.assertIn("Only HTTP and HTTPS URLs are allowed", result.errors)
    
    def test_url_validation_local_address(self):
        """Test validation blocks local addresses"""
        result = self.validator.validate_url("http://localhost/malicious")
        self.assertFalse(result.is_valid)
        self.assertEqual(result.risk_level, "high")
    
    def test_text_sanitization(self):
        """Test text sanitization functionality"""
        dirty_text = "<script>alert('test')</script> & 'quotes'"
        result = self.validator.validate_text_input(dirty_text)
        if result.sanitized_input:
            # Should not contain raw script tags
            self.assertNotIn("<script>", result.sanitized_input)
            # Should be HTML escaped
            self.assertIn("&lt;", result.sanitized_input)


class TestRateLimitManager(unittest.TestCase):
    """Test rate limiting functionality"""
    
    def setUp(self):
        self.rate_limiter = RateLimitManager()
    
    def test_first_request_allowed(self):
        """Test that first request is allowed"""
        allowed, message = self.rate_limiter.is_allowed("test_user", "test_endpoint")
        self.assertTrue(allowed)
        self.assertEqual(message, "Request allowed")
    
    def test_rate_limit_enforcement(self):
        """Test that rate limits are enforced"""
        user_id = "heavy_user"
        endpoint = "analyze"
        
        # Make requests up to the limit
        limits = self.rate_limiter._get_rate_limits(endpoint)
        for _ in range(limits['requests_per_minute']):
            allowed, _ = self.rate_limiter.is_allowed(user_id, endpoint)
            self.assertTrue(allowed)
        
        # Next request should be denied
        allowed, message = self.rate_limiter.is_allowed(user_id, endpoint)
        self.assertFalse(allowed)
        self.assertIn("Rate limit exceeded", message)
    
    def test_different_endpoints_separate_limits(self):
        """Test that different endpoints have separate limits"""
        user_id = "test_user"
        
        # Use up limit for one endpoint
        limits1 = self.rate_limiter._get_rate_limits("analyze")
        for _ in range(limits1['requests_per_minute']):
            self.rate_limiter.is_allowed(user_id, "analyze")
        
        # Should still be allowed for different endpoint
        allowed, _ = self.rate_limiter.is_allowed(user_id, "batch_process")
        self.assertTrue(allowed)
    
    def test_ip_blocking(self):
        """Test IP blocking for excessive requests"""
        user_id = "malicious_user"
        endpoint = "analyze"
        
        # Make excessive requests to trigger blocking
        limits = self.rate_limiter._get_rate_limits(endpoint)
        excessive_requests = limits['requests_per_minute'] * 3
        
        for _ in range(excessive_requests):
            self.rate_limiter.is_allowed(user_id, endpoint)
        
        # User should now be blocked
        self.assertIn(user_id, self.rate_limiter.blocked_ips)


class TestSecureSession(unittest.TestCase):
    """Test secure session management"""
    
    def setUp(self):
        self.session_manager = SecureSession(session_timeout=60)
    
    def test_create_session(self):
        """Test session creation"""
        session_id = self.session_manager.create_session("user123")
        self.assertIsInstance(session_id, str)
        self.assertGreater(len(session_id), 20)  # Should be long random string
    
    def test_validate_valid_session(self):
        """Test validation of valid session"""
        user_id = "user123"
        session_id = self.session_manager.create_session(user_id)
        
        session_data = self.session_manager.validate_session(session_id)
        self.assertIsNotNone(session_data)
        self.assertEqual(session_data['user_id'], user_id)
        self.assertIn('csrf_token', session_data)
    
    def test_validate_invalid_session(self):
        """Test validation of invalid session"""
        result = self.session_manager.validate_session("invalid_session_id")
        self.assertIsNone(result)
    
    def test_session_expiration(self):
        """Test session expiration"""
        # Create session with very short timeout
        short_session = SecureSession(session_timeout=0.1)
        session_id = short_session.create_session("user123")
        
        # Wait for expiration
        import time
        time.sleep(0.2)
        
        result = short_session.validate_session(session_id)
        self.assertIsNone(result)
    
    def test_session_invalidation(self):
        """Test manual session invalidation"""
        session_id = self.session_manager.create_session("user123")
        
        # Validate session exists
        self.assertIsNotNone(self.session_manager.validate_session(session_id))
        
        # Invalidate session
        self.session_manager.invalidate_session(session_id)
        
        # Session should no longer be valid
        self.assertIsNone(self.session_manager.validate_session(session_id))


class TestDataAnonymizer(unittest.TestCase):
    """Test data anonymization utilities"""
    
    def test_hash_identifier(self):
        """Test identifier hashing"""
        identifier = "user@example.com"
        hash1 = DataAnonymizer.hash_identifier(identifier)
        hash2 = DataAnonymizer.hash_identifier(identifier)
        
        # Same input should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Hash should be different from original
        self.assertNotEqual(hash1, identifier)
        
        # Should be fixed length
        self.assertEqual(len(hash1), 16)
    
    def test_redact_personal_info(self):
        """Test personal information redaction"""
        text = "Contact me at john.doe@example.com or call 555-123-4567. Visit https://example.com"
        redacted = DataAnonymizer.redact_personal_info(text)
        
        # Should not contain original email, phone, or URL
        self.assertNotIn("john.doe@example.com", redacted)
        self.assertNotIn("555-123-4567", redacted)
        self.assertNotIn("https://example.com", redacted)
        
        # Should contain redaction markers
        self.assertIn("[EMAIL]", redacted)
        self.assertIn("[PHONE]", redacted)
        self.assertIn("[URL]", redacted)


class TestSecurityUtilities(unittest.TestCase):
    """Test security utility functions"""
    
    def test_secure_hash(self):
        """Test secure hashing function"""
        data = "sensitive_data"
        hash1 = secure_hash(data)
        hash2 = secure_hash(data)
        
        # Same input should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different input should produce different hash
        hash3 = secure_hash("different_data")
        self.assertNotEqual(hash1, hash3)
        
        # Should be SHA256 length
        self.assertEqual(len(hash1), 64)
    
    def test_global_validate_input(self):
        """Test global input validation function"""
        result = validate_input("valid text")
        self.assertTrue(result.is_valid)
        
        result = validate_input("<script>alert('xss')</script>")
        self.assertFalse(result.is_valid)
    
    def test_global_rate_limit_check(self):
        """Test global rate limit check function"""
        allowed, message = check_rate_limit("test_user", "default")
        self.assertTrue(allowed)


if __name__ == '__main__':
    unittest.main()
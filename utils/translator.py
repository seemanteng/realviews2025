"""
Multilingual translation utilities for RealViews
Supports translation and language detection using Google Translate API
"""

import logging
from typing import Dict, Optional, Tuple
from deep_translator import GoogleTranslator
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualTranslator:
    """Free multilingual translator using Google Translate"""
    
    def __init__(self):
        self.translator = GoogleTranslator()
        
        # Supported languages for review analysis (using deep-translator codes)
        self.supported_languages = {
            'en': 'English',
            'zh-CN': 'Chinese (Simplified)',
            'zh-TW': 'Chinese (Traditional)',
            'es': 'Spanish', 
            'ms': 'Malay',
            'ja': 'Japanese',
            'ko': 'Korean',
            'de': 'German',
            'fr': 'French',
            'hi': 'Hindi',
            'ta': 'Tamil',
            'ar': 'Arabic',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'it': 'Italian',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian'
        }
        
        # Mapping from pattern detection to deep-translator codes
        self.pattern_to_lang_code = {
            'zh': 'zh-CN',  # Chinese simplified
            'ja': 'ja',     # Japanese
            'ko': 'ko',     # Korean  
            'ar': 'ar',     # Arabic
            'hi': 'hi',     # Hindi
            'ta': 'ta',     # Tamil
            'th': 'th',     # Thai
            'ru': 'ru',     # Russian
        }
        
        # Common non-English patterns for quick detection (order matters!)
        self.language_patterns = {
            'ja': r'[\u3040-\u309f\u30a0-\u30ff]',  # Japanese hiragana/katakana (check first)
            'ko': r'[\uac00-\ud7af]',  # Korean
            'zh': r'[\u4e00-\u9fff]',  # Chinese characters (check after Japanese/Korean)
            'ar': r'[\u0600-\u06ff]',  # Arabic
            'hi': r'[\u0900-\u097f]',  # Hindi/Devanagari
            'ta': r'[\u0b80-\u0bff]',  # Tamil
            'th': r'[\u0e00-\u0e7f]',  # Thai
            'ru': r'[\u0400-\u04ff]',  # Cyrillic
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of the text
        Returns: (language_code, confidence)
        """
        try:
            # Quick pattern-based detection for non-Latin scripts
            for pattern_lang, pattern in self.language_patterns.items():
                if re.search(pattern, text):
                    # Map pattern language to deep-translator code
                    lang_code = self.pattern_to_lang_code.get(pattern_lang, pattern_lang)
                    return lang_code, 0.9
            
            # If no patterns match, assume English for Latin scripts
            return 'en', 0.7
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en', 0.5  # Default to English
    
    def translate_text(self, text: str, target_lang: str = 'en', source_lang: str = None) -> Dict:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_lang: Target language code (default: 'en')
            source_lang: Source language code (auto-detect if None)
        
        Returns:
            Dict with translation info
        """
        try:
            # Handle empty or very short text
            if not text or len(text.strip()) < 2:
                return {
                    'original_text': text,
                    'translated_text': text,
                    'source_language': 'unknown',
                    'target_language': target_lang,
                    'source_language_name': 'Unknown',
                    'confidence': 0.0,
                    'translation_needed': False,
                    'error': 'Text too short for translation'
                }
            
            # Detect source language if not provided
            if source_lang is None:
                detected_lang, confidence = self.detect_language(text)
                source_lang = detected_lang
            else:
                confidence = 1.0
            
            # Skip translation if already in target language or unknown
            if source_lang == target_lang or source_lang == 'en':
                return {
                    'original_text': text,
                    'translated_text': text,
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'source_language_name': self.supported_languages.get(source_lang, 'English'),
                    'confidence': confidence,
                    'translation_needed': False
                }
            
            # Ensure source language is supported
            if source_lang not in self.supported_languages:
                logger.warning(f"Unsupported source language: {source_lang}")
                return {
                    'original_text': text,
                    'translated_text': text,
                    'source_language': source_lang,
                    'target_language': target_lang,
                    'source_language_name': f'Unknown ({source_lang})',
                    'confidence': 0.0,
                    'translation_needed': False,
                    'error': f'Unsupported language: {source_lang}'
                }
            
            # Perform translation using deep-translator with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    translator = GoogleTranslator(source=source_lang, target=target_lang)
                    translated_text = translator.translate(text)
                    
                    # Validate translation result
                    if not translated_text or translated_text.strip() == "":
                        raise ValueError("Empty translation result")
                    
                    # Check if translation actually occurred (sometimes returns same text)
                    actual_translation_needed = translated_text.lower().strip() != text.lower().strip()
                    
                    return {
                        'original_text': text,
                        'translated_text': translated_text,
                        'source_language': source_lang,
                        'target_language': target_lang,
                        'source_language_name': self.supported_languages.get(source_lang, 'Unknown'),
                        'confidence': confidence,
                        'translation_needed': actual_translation_needed
                    }
                    
                except Exception as inner_e:
                    if attempt == max_retries - 1:  # Last attempt
                        raise inner_e
                    logger.warning(f"Translation attempt {attempt + 1} failed: {inner_e}, retrying...")
                    import time
                    time.sleep(0.5)  # Brief pause before retry
            
        except Exception as e:
            logger.error(f"Translation failed after retries: {e}")
            # Fallback - return original text
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': source_lang or 'unknown',
                'target_language': target_lang,
                'source_language_name': self.supported_languages.get(source_lang, 'Unknown'),
                'confidence': 0.0,
                'translation_needed': False,
                'error': str(e)
            }
    
    def translate_for_analysis(self, text: str) -> Dict:
        """
        Translate text to English for analysis while preserving original
        """
        return self.translate_text(text, target_lang='en')
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes and names"""
        return self.supported_languages.copy()
    
    def is_supported_language(self, lang_code: str) -> bool:
        """Check if language is supported"""
        return lang_code in self.supported_languages
    
    def batch_translate(self, texts: list, target_lang: str = 'en') -> list:
        """
        Translate multiple texts
        
        Args:
            texts: List of texts to translate
            target_lang: Target language code
        
        Returns:
            List of translation results
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.translate_text(text, target_lang)
                result['index'] = i
                results.append(result)
                
                # Add small delay to avoid rate limiting
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to translate text {i}: {e}")
                results.append({
                    'index': i,
                    'original_text': text,
                    'translated_text': text,
                    'error': str(e),
                    'translation_needed': False
                })
        
        return results

# Global translator instance
_translator_instance = None

def get_translator() -> MultilingualTranslator:
    """Get global translator instance"""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = MultilingualTranslator()
    return _translator_instance

def translate_review(text: str, target_lang: str = 'en') -> Dict:
    """Convenience function to translate a review"""
    translator = get_translator()
    return translator.translate_for_analysis(text)

def detect_review_language(text: str) -> Tuple[str, float]:
    """Convenience function to detect review language"""
    translator = get_translator()
    return translator.detect_language(text)
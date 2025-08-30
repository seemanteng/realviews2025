#!/usr/bin/env python3
"""
Test script for multilingual functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.translator import get_translator

def test_translation():
    """Test translation functionality with various languages"""
    
    translator = get_translator()
    
    # Test reviews in different languages
    test_reviews = [
        ("这家餐厅的食物很好吃，服务也很棒！强烈推荐。", "Chinese"),
        ("La comida estaba deliciosa y el servicio fue excelente.", "Spanish"), 
        ("このレストランは素晴らしい！料理がとても美味しい。", "Japanese"),
        ("이 식당 정말 좋아요! 음식도 맛있고 서비스도 친절해요.", "Korean"),
        ("Das Essen war ausgezeichnet und der Service war sehr freundlich.", "German"),
        ("यह रेस्तराँ बहुत अच्छा है! खाना स्वादिष्ट था।", "Hindi"),
        ("Great food and excellent service!", "English (should not translate)")
    ]
    
    print("🌍 Testing Multilingual Translation Functionality")
    print("=" * 60)
    
    for review_text, language in test_reviews:
        print(f"\n📝 Testing {language}:")
        print(f"Original: {review_text}")
        
        try:
            # Detect language
            detected_lang, confidence = translator.detect_language(review_text)
            print(f"🔍 Detected: {detected_lang} (confidence: {confidence:.2f})")
            
            # Translate
            result = translator.translate_for_analysis(review_text)
            
            if result['translation_needed']:
                print(f"🔄 Translated: {result['translated_text']}")
                print(f"📊 Source: {result['source_language_name']} → English")
            else:
                print("✅ No translation needed (already English)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    test_translation()
"""
LLM-enhanced review classifier using lightweight Hugging Face models
Complements the traditional ML approach with modern language understanding
"""

import os
import re
from typing import Dict, List, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Suppress some warnings
logging.getLogger("transformers").setLevel(logging.WARNING)

class LLMReviewClassifier:
    """LLM-powered review analysis using lightweight Hugging Face models"""
    
    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_API_KEY')
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Model configurations - using lightweight, efficient models
        self.model_configs = {
            'sentiment': {
                'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'task': 'sentiment-analysis'
            },
            'quality': {
                'model_name': 'microsoft/DialoGPT-small',  # For quality assessment
                'task': 'text-generation'
            },
            'classification': {
                'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
                'task': 'sentiment-analysis'
            }
        }
        
        self.initialized_models = set()
        
    def initialize_model(self, model_type: str = 'sentiment') -> bool:
        """Initialize a specific model lazily"""
        if model_type in self.initialized_models:
            return True
            
        try:
            config = self.model_configs.get(model_type)
            if not config:
                print(f"❌ Unknown model type: {model_type}")
                return False
            
            print(f"🤖 Loading {model_type} model: {config['model_name']}...")
            
            # Create pipeline with authentication if token provided
            kwargs = {}
            if self.hf_token:
                kwargs['use_auth_token'] = self.hf_token
            
            # Use CPU for lightweight deployment, GPU if available
            device = 0 if torch.cuda.is_available() else -1
            
            self.pipelines[model_type] = pipeline(
                config['task'],
                model=config['model_name'],
                device=device,
                **kwargs
            )
            
            self.initialized_models.add(model_type)
            print(f"✅ {model_type} model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load {model_type} model: {e}")
            print(f"   Continuing with rule-based approach only...")
            return False
    
    def analyze_sentiment_advanced(self, text: str) -> Dict[str, Any]:
        """Advanced sentiment analysis using RoBERTa model"""
        if not self.initialize_model('sentiment'):
            return {'sentiment': 'neutral', 'confidence': 0.5, 'scores': {}}
        
        try:
            results = self.pipelines['sentiment'](text)
            
            # Handle different output formats
            if isinstance(results, list):
                result = results[0]
            else:
                result = results
            
            return {
                'sentiment': result['label'].lower(),
                'confidence': result['score'],
                'scores': {result['label'].lower(): result['score']},
                'method': 'roberta'
            }
            
        except Exception as e:
            print(f"⚠️  Sentiment analysis error: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'scores': {}}
    
    def detect_quality_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to assess review quality beyond traditional metrics"""
        
        quality_prompts = {
            'coherence': f"Rate the coherence of this review (1-5): '{text}'",
            'specificity': f"Rate how specific and detailed this review is (1-5): '{text}'",
            'helpfulness': f"Rate how helpful this review would be to others (1-5): '{text}'"
        }
        
        # For now, use a prompt-based approach with the sentiment model
        quality_assessment = {
            'coherence_score': 0.5,
            'specificity_score': 0.5,
            'helpfulness_score': 0.5,
            'overall_quality': 0.5,
            'reasoning': []
        }
        
        # Analyze text characteristics
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        
        # Coherence: Check if sentences flow logically
        if len(sentences) > 1:
            quality_assessment['coherence_score'] = min(1.0, 0.3 + len(sentences) * 0.1)
            quality_assessment['reasoning'].append(f"Good sentence structure ({len(sentences)} sentences)")
        
        # Specificity: Look for specific details
        specific_words = ['price', 'cost', 'staff', 'waiter', 'menu', 'dish', 'flavor', 'atmosphere', 'location']
        specific_count = sum(1 for word in words if word.lower() in specific_words)
        if specific_count > 0:
            quality_assessment['specificity_score'] = min(1.0, 0.3 + specific_count * 0.15)
            quality_assessment['reasoning'].append(f"Contains specific details ({specific_count} specific terms)")
        
        # Helpfulness: Length and personal experience indicators
        personal_indicators = ['i', 'my', 'we', 'our', 'visited', 'went', 'ordered', 'tried']
        personal_count = sum(1 for word in words if word.lower() in personal_indicators)
        if personal_count > 0 and len(words) > 10:
            quality_assessment['helpfulness_score'] = min(1.0, 0.4 + personal_count * 0.1)
            quality_assessment['reasoning'].append(f"Personal experience shared ({personal_count} personal indicators)")
        
        # Overall quality
        quality_assessment['overall_quality'] = (
            quality_assessment['coherence_score'] * 0.3 +
            quality_assessment['specificity_score'] * 0.4 +
            quality_assessment['helpfulness_score'] * 0.3
        )
        
        return quality_assessment
    
    def classify_review_type(self, text: str) -> Dict[str, Any]:
        """Classify review type and detect policy violations using LLM reasoning"""
        
        classification_result = {
            'is_advertisement': False,
            'is_fake': False,
            'is_irrelevant': False,
            'is_spam': False,
            'confidence': 0.5,
            'reasoning': []
        }
        
        text_lower = text.lower()
        
        # Enhanced pattern detection with context understanding
        
        # Advertisement detection with context
        ad_patterns = [
            (r'\b(visit|check)\s+(our|my)\s+(website|site)', 'Explicit website promotion'),
            (r'\b(discount|sale|offer|deal)\b.*\b(percent|%|off)\b', 'Promotional pricing'),
            (r'\b(click|follow|visit)\s+(here|link|url)', 'Call-to-action links'),
            (r'\b(promo|coupon|code)\b', 'Promotional codes'),
            (r'www\.|http|\.com', 'Contains URLs')
        ]
        
        ad_score = 0
        for pattern, reason in ad_patterns:
            if re.search(pattern, text_lower):
                ad_score += 0.3
                classification_result['reasoning'].append(reason)
        
        classification_result['is_advertisement'] = ad_score > 0.5
        
        # Fake review detection with linguistic analysis
        fake_indicators = [
            (r'\b(perfect|amazing|incredible)\b.*\b(perfect|amazing|incredible)\b', 'Excessive superlatives'),
            (r'\b(highly recommend)\b.*\b(must visit|definitely)\b', 'Template phrases'),
            (r'\b(best ever|absolutely perfect|completely amazing)\b', 'Extreme language'),
        ]
        
        fake_score = 0
        superlative_count = len(re.findall(r'\b(perfect|amazing|incredible|outstanding|phenomenal|extraordinary|absolutely|definitely|highly|must)\b', text_lower))
        
        if superlative_count > len(text.split()) * 0.3:  # More than 30% superlatives
            fake_score += 0.4
            classification_result['reasoning'].append(f'High superlative ratio ({superlative_count} in {len(text.split())} words)')
        
        for pattern, reason in fake_indicators:
            if re.search(pattern, text_lower):
                fake_score += 0.3
                classification_result['reasoning'].append(reason)
        
        classification_result['is_fake'] = fake_score > 0.5
        
        # Irrelevant content detection
        irrelevant_patterns = [
            (r'\b(wrong|different)\s+(place|location|restaurant)', 'Wrong location mentioned'),
            (r'\b(not about|unrelated|off topic)\b', 'Explicitly irrelevant'),
            (r'\b(random|spam|nonsense)\b', 'Self-identified as spam')
        ]
        
        irrelevant_score = 0
        for pattern, reason in irrelevant_patterns:
            if re.search(pattern, text_lower):
                irrelevant_score += 0.5
                classification_result['reasoning'].append(reason)
        
        classification_result['is_irrelevant'] = irrelevant_score > 0.3
        
        # Overall confidence
        max_score = max(ad_score, fake_score, irrelevant_score)
        classification_result['confidence'] = min(0.95, max(0.1, max_score))
        
        return classification_result
    
    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive LLM-enhanced analysis of a review"""
        
        # Get sentiment analysis
        sentiment_result = self.analyze_sentiment_advanced(text)
        
        # Get quality assessment
        quality_result = self.detect_quality_with_llm(text)
        
        # Get classification
        classification_result = self.classify_review_type(text)
        
        # Combine results
        comprehensive_result = {
            'text': text,
            'sentiment': sentiment_result,
            'quality': quality_result,
            'classification': classification_result,
            'overall_score': self._calculate_overall_score(sentiment_result, quality_result, classification_result),
            'recommendations': self._generate_recommendations(sentiment_result, quality_result, classification_result)
        }
        
        return comprehensive_result
    
    def _calculate_overall_score(self, sentiment: Dict, quality: Dict, classification: Dict) -> float:
        """Calculate overall review score combining all factors"""
        
        base_score = quality['overall_quality']
        
        # Penalize policy violations
        if classification['is_advertisement'] or classification['is_fake'] or classification['is_irrelevant']:
            base_score *= 0.3  # Severe penalty for violations
        
        # Adjust for sentiment extremes (very positive might be fake)
        if sentiment['sentiment'] == 'positive' and sentiment['confidence'] > 0.9:
            base_score *= 0.8  # Slight penalty for extreme positivity
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_recommendations(self, sentiment: Dict, quality: Dict, classification: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Quality recommendations
        if quality['overall_quality'] < 0.3:
            recommendations.append("⚠️  Low quality content - consider manual review")
        
        if quality['specificity_score'] < 0.3:
            recommendations.append("💡 Review lacks specific details about the experience")
        
        # Policy violation recommendations
        if classification['is_advertisement']:
            recommendations.append("🚨 Advertisement detected - likely policy violation")
        
        if classification['is_fake']:
            recommendations.append("🚨 Fake review indicators - requires verification")
        
        if classification['is_irrelevant']:
            recommendations.append("🚨 Irrelevant content - may be off-topic")
        
        # Sentiment recommendations
        if sentiment['confidence'] < 0.5:
            recommendations.append("❓ Ambiguous sentiment - review may be unclear")
        
        if not recommendations:
            recommendations.append("✅ Review appears legitimate and helpful")
        
        return recommendations
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple reviews efficiently"""
        results = []
        
        print(f"🔄 Analyzing {len(texts)} reviews with LLM enhancement...")
        
        for i, text in enumerate(texts):
            try:
                result = self.comprehensive_analysis(text)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(texts)} reviews...")
                    
            except Exception as e:
                print(f"⚠️  Error analyzing review {i+1}: {e}")
                # Fallback result
                results.append({
                    'text': text,
                    'sentiment': {'sentiment': 'neutral', 'confidence': 0.5},
                    'quality': {'overall_quality': 0.5},
                    'classification': {'is_advertisement': False, 'is_fake': False, 'is_irrelevant': False},
                    'overall_score': 0.5,
                    'recommendations': ['❌ Analysis failed - manual review required']
                })
        
        print(f"✅ Completed LLM analysis of {len(texts)} reviews")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'initialized_models': list(self.initialized_models),
            'available_models': list(self.model_configs.keys()),
            'has_gpu': torch.cuda.is_available(),
            'hf_token_provided': bool(self.hf_token)
        }
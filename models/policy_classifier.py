import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from utils.data_processing import ReviewProcessor
from utils.user_data_analyzer import UserDataAnalyzer

# Translation support - optional
try:
    from utils.translator import get_translator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("Translation not available. Install 'deep-translator' for multilingual support.")

# LLM integration
try:
    from models.llm_classifier import LLMReviewClassifier
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM classifier not available. Continuing with traditional ML only.")

class PolicyClassifier:
    """ML-based policy violation classifier with ensemble approach"""
    
    def __init__(self, hf_token: str = None, use_llm: bool = True):
        self.processor = ReviewProcessor()
        self.user_analyzer = UserDataAnalyzer()
        self.models = {}
        self.vectorizers = {}
        self.is_trained = False
        self.use_llm = use_llm and LLM_AVAILABLE
        self.user_suspicion_weights = {
            'high_suspicion_account': 0.3,
            'burst_activity': 0.2, 
            'composite_score': 0.15
        }
        
        # Initialize LLM classifier if available
        if self.use_llm:
            try:
                self.llm_classifier = LLMReviewClassifier(hf_token=hf_token)
                print("LLM classifier initialized successfully")
            except Exception as e:
                print(f"Failed to initialize LLM classifier: {e}")
                self.use_llm = False
                self.llm_classifier = None
        else:
            self.llm_classifier = None
        
        # Initialize rule-based components
        self._init_rule_patterns()
        
        # Create synthetic training data for demonstration
        self._create_demo_models()
    
    def _init_rule_patterns(self):
        """Initialize rule-based detection patterns"""
        self.advertisement_patterns = [
            # English patterns
            r'\b(discount|sale|offer|deal|coupon|promo)\b',
            r'\b(buy now|click here|visit our|website)\b',
            r'\b(free shipping|limited time|special offer)\b',
            r'(http|www|\.com|\.net|\.org)',
            r'\b\d+%\s*off\b',
            r'\$\d+.*\b(off|discount)\b',
            
            # Chinese patterns - more specific to avoid false positives
            r'(优惠.*活动|折扣.*活动|促销.*活动|特价.*活动)',  # Promotional activities
            r'(关注.*获得|扫码.*获得|添加.*获得)',            # Follow/scan/add to get something
            r'(免费.*获得|免费.*领取|送.*礼品|赠.*奖品)',      # Free gifts/prizes (promotional context)
            r'(联系.*电话|联系.*微信|联系.*QQ)',              # Contact info for promotion
            r'(转发.*活动|分享.*获得|点赞.*奖励)',            # Social media promotional actions
            r'(访问.*网站|点击.*链接|下载.*APP)',             # Visit website/click link/download app
            r'(立即.*购买|马上.*下单|现在.*订购)',             # Urgent purchase language
            r'\d+折.*优惠',                                 # Discount offers
            r'满\d+.*减\d+',                               # Spend X get Y off
        ]
        
        self.irrelevant_patterns = [
            # English patterns
            r'\b(wrong place|different location|not about)\b',
            r'\b(irrelevant|unrelated|off topic)\b',
            r'\b(random|spam|nonsense)\b',
            
            # Chinese patterns
            r'(不是|不对|错误|不相关)',              # Not, wrong, incorrect, unrelated
            r'(别的|其他|另一个)',                  # Other, another, different
            r'(搞错|弄错|找错)',                    # Made mistake, got wrong
        ]
        
        self.fake_review_indicators = [
            # English patterns
            r'\b(perfect|amazing|best ever|incredible)\b.*\b(perfect|amazing|best ever|incredible)\b',
            r'\b(highly recommend)\b.*\b(must visit|definitely)\b',
            r'\b(outstanding|phenomenal|extraordinary)\b.*\b(absolutely|definitely)\b',
            
            # Chinese patterns
            r'(非常.*非常.*非常|很.*很.*很.*很)',      # Excessive "very" repetition
            r'(棒.*棒.*棒|好.*好.*好.*好)',          # Excessive "good/great" repetition  
            r'(推荐.*推荐.*推荐|赞.*赞.*赞)',        # Excessive "recommend/praise" repetition
            r'(完美.*完美|极好.*极好)',              # Excessive "perfect/excellent"
            r'！{3,}',                             # Multiple Chinese exclamation marks
        ]
    
    def _create_demo_models(self):
        """Create demonstration models with synthetic data for hackathon"""
        # Generate synthetic training data
        synthetic_data = self._generate_synthetic_data()
        
        # Train models on synthetic data
        self._train_models(synthetic_data)
        
        self.is_trained = True
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic training data for demonstration"""
        
        # Clean reviews (no violations)
        clean_reviews = [
            "Great food and excellent service. The pasta was perfectly cooked.",
            "Nice atmosphere, friendly staff. Had the salmon which was delicious.",
            "Good location, easy parking. The pizza was okay, nothing special.",
            "Decent restaurant with reasonable prices. Service could be faster.",
            "Beautiful interior design. The dessert was amazing!",
            "Family-friendly place with good portion sizes.",
            "Fresh ingredients, well-prepared dishes. Will come back.",
            "Cozy ambiance, perfect for date night. Food was good quality.",
            "Quick service, clean environment. The burgers were tasty.",
            "Traditional cuisine, authentic flavors. Worth trying."
        ]
        
        # Advertisement violations
        ad_violations = [
            "Amazing restaurant! Visit our website www.example.com for 50% off!",
            "Best deals in town! Click here for special discounts and offers.",
            "Free shipping on orders over $25! Use coupon code SAVE50 now!",
            "Limited time offer! Buy now and get 30% off your next meal!",
            "Don't miss our sale! Visit our store for incredible deals today!",
            "Promo alert: Get free dessert with every meal this week only!",
            "Special discount for new customers - check out our website!",
            "Amazing prices! Follow the link for exclusive restaurant deals!"
        ]
        
        # Irrelevant content
        irrelevant_reviews = [
            "This review is about a completely different restaurant in another city.",
            "Wrong place! I was looking for the hotel, not this restaurant.",
            "Off topic: Does anyone know how to fix my car? Random question.",
            "This is irrelevant content that has nothing to do with dining.",
            "Unrelated post about my vacation photos and travel experiences.",
            "Random spam content with no connection to restaurant reviews.",
            "Not about this location - talking about a different business entirely.",
            "Irrelevant comment about weather and traffic conditions today."
        ]
        
        # Fake reviews
        fake_reviews = [
            "Perfect restaurant! Absolutely perfect food! Everything is perfect! Must visit!",
            "Amazing incredible outstanding phenomenal experience! Definitely highly recommend!",
            "Best ever best ever! Extraordinary service extraordinary food extraordinary everything!",
            "Absolutely amazing definitely perfect incredible outstanding must visit highly recommend!",
            "Perfect amazing incredible best restaurant ever! Definitely must visit absolutely!",
            "Outstanding phenomenal extraordinary perfect experience! Highly recommend definitely!",
            "Amazing perfect incredible restaurant! Best ever! Absolutely must visit definitely!",
            "Perfect service perfect food perfect everything! Amazing incredible outstanding!"
        ]
        
        # Create DataFrame
        data = []
        
        # Add clean reviews
        for review in clean_reviews:
            data.append({
                'text': review,
                'has_violation': False,
                'violation_type': 'none',
                'advertisement': False,
                'irrelevant': False,
                'fake': False
            })
        
        # Add advertisement violations  
        for review in ad_violations:
            data.append({
                'text': review,
                'has_violation': True,
                'violation_type': 'advertisement',
                'advertisement': True,
                'irrelevant': False,
                'fake': False
            })
        
        # Add irrelevant content
        for review in irrelevant_reviews:
            data.append({
                'text': review,
                'has_violation': True,
                'violation_type': 'irrelevant',
                'advertisement': False,
                'irrelevant': True,
                'fake': False
            })
        
        # Add fake reviews
        for review in fake_reviews:
            data.append({
                'text': review,
                'has_violation': True,
                'violation_type': 'fake',
                'advertisement': False,
                'irrelevant': False,
                'fake': True
            })
        
        return pd.DataFrame(data)
    
    def _train_models(self, data: pd.DataFrame):
        """Train ensemble of classifiers"""
        
        # Prepare features
        X_text = data['text'].values
        
        # TF-IDF vectorization
        self.vectorizers['tfidf'] = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X_tfidf = self.vectorizers['tfidf'].fit_transform(X_text)
        
        # Extract manual features
        feature_data = []
        for text in X_text:
            features = self.processor.extract_features(text)
            feature_data.append([
                features.get('word_count', 0),
                features.get('promotional_keywords', 0),
                features.get('fake_indicators', 0),
                features.get('irrelevant_indicators', 0),
                features.get('sentiment_compound', 0) if 'sentiment_compound' in features else features.get('sentiment_polarity', 0),
                features.get('uppercase_ratio', 0),
                features.get('punctuation_ratio', 0)
            ])
        
        X_manual = np.array(feature_data)
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, X_manual])
        
        # Train main violation classifier
        y_violation = data['has_violation'].values
        self.models['violation'] = LogisticRegression(random_state=42)
        self.models['violation'].fit(X_combined, y_violation)
        
        # Train specific violation type classifiers
        for violation_type in ['advertisement', 'irrelevant', 'fake']:
            y_type = data[violation_type].values
            self.models[violation_type] = RandomForestClassifier(n_estimators=50, random_state=42)
            self.models[violation_type].fit(X_combined, y_type)
    
    def _extract_features_for_prediction(self, text: str) -> np.ndarray:
        """Extract features for a single text sample"""
        # TF-IDF features
        X_tfidf = self.vectorizers['tfidf'].transform([text])
        
        # Manual features
        features = self.processor.extract_features(text)
        manual_features = np.array([[
            features.get('word_count', 0),
            features.get('promotional_keywords', 0),
            features.get('fake_indicators', 0),
            features.get('irrelevant_indicators', 0),
            features.get('sentiment_compound', 0) if 'sentiment_compound' in features else features.get('sentiment_polarity', 0),
            features.get('uppercase_ratio', 0),
            features.get('punctuation_ratio', 0)
        ]])
        
        # Combine features
        from scipy.sparse import hstack
        X_combined = hstack([X_tfidf, manual_features])
        
        return X_combined
    
    def _rule_based_detection(self, text: str, product_info: str = None, context_analysis: Dict = None) -> Dict[str, Any]:
        """Apply rule-based detection with optional context awareness"""
        text_lower = text.lower()
        
        results = {
            'advertisement': {
                'detected': False,
                'confidence': 0.0,
                'reason': ''
            },
            'irrelevant': {
                'detected': False,
                'confidence': 0.0,
                'reason': ''
            },
            'fake': {
                'detected': False,
                'confidence': 0.0,
                'reason': ''
            }
        }
        
        # Advertisement detection
        ad_matches = []
        for pattern in self.advertisement_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                ad_matches.extend(matches)
        
        if ad_matches:
            results['advertisement']['detected'] = True
            results['advertisement']['confidence'] = min(1.0, len(ad_matches) * 0.3)
            results['advertisement']['reason'] = f"Contains promotional content: {', '.join(set(ad_matches))}"
        
        # Irrelevant content detection (enhanced with context)
        irrelevant_matches = []
        for pattern in self.irrelevant_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                irrelevant_matches.extend(matches)
        
        # Add context-based irrelevance detection
        if context_analysis:
            context_violations = context_analysis.get('context_violations', {})
            if context_violations.get('irrelevant_content') or context_violations.get('context_mismatch'):
                irrelevant_matches.extend(['context_mismatch'])
                context_reason = ', '.join(context_violations.get('details', []))
                if context_reason:
                    irrelevant_matches.append(f"({context_reason})")
        
        if irrelevant_matches:
            results['irrelevant']['detected'] = True
            results['irrelevant']['confidence'] = min(1.0, len(irrelevant_matches) * 0.4)
            results['irrelevant']['reason'] = f"Contains irrelevant indicators: {', '.join(set(irrelevant_matches))}"
        
        # Fake review detection
        fake_matches = []
        for pattern in self.fake_review_indicators:
            if re.search(pattern, text_lower):
                fake_matches.append(pattern.split('|')[0].strip(r'\b()'))
        
        # Check for excessive superlatives
        superlatives = ['perfect', 'amazing', 'incredible', 'outstanding', 'phenomenal', 'extraordinary']
        superlative_count = sum(1 for word in superlatives if word in text_lower)
        
        if superlative_count > 2 or fake_matches:
            results['fake']['detected'] = True
            results['fake']['confidence'] = min(1.0, superlative_count * 0.2 + len(fake_matches) * 0.3)
            results['fake']['reason'] = f"Excessive positive language detected (superlatives: {superlative_count})"
        
        return results
    
    def predict_single(self, text: str, product_info: str = None, user_metadata: Dict[str, Any] = None, analyze_original: bool = False) -> Dict[str, Any]:
        """Predict policy violations for a single review with optional product context"""
        
        # Translation support (optional)
        translation_result = None
        analysis_text = text  # Default to original text
        original_text = text
        
        if TRANSLATION_AVAILABLE:
            try:
                translator = get_translator()
                translation_result = translator.translate_for_analysis(text)
                
                # Choose what text to analyze based on mode
                if analyze_original:
                    analysis_text = translation_result['original_text']  # Analyze original
                else:
                    analysis_text = translation_result['translated_text']  # Analyze translation
                    
                original_text = translation_result['original_text']
            except Exception as e:
                print(f"Translation failed: {e}")
        
        # Preprocess text
        cleaned_text = self.processor.preprocess_text(analysis_text)
        
        # LLM-enhanced analysis if available
        llm_results = None
        if self.use_llm and self.llm_classifier:
            try:
                llm_results = self.llm_classifier.comprehensive_analysis(text)
            except Exception as e:
                print(f"LLM analysis failed: {e}")
                llm_results = None
        
        # Context-aware analysis if product info provided
        context_analysis = None
        if product_info:
            context_analysis = self.processor.analyze_with_context(text, product_info)
        
        # Rule-based detection (enhanced with context)
        rule_results = self._rule_based_detection(text, product_info, context_analysis)
        
        # ML-based detection (if models are trained)
        ml_results = {}
        if self.is_trained:
            try:
                X = self._extract_features_for_prediction(cleaned_text)
                
                # Main violation prediction
                violation_prob = self.models['violation'].predict_proba(X)[0, 1]
                has_ml_violation = violation_prob > 0.5
                
                # Specific violation predictions
                ml_results = {}
                for violation_type in ['advertisement', 'irrelevant', 'fake']:
                    type_prob = self.models[violation_type].predict_proba(X)[0, 1]
                    ml_results[violation_type] = {
                        'detected': type_prob > 0.5,
                        'confidence': type_prob,
                        'reason': f'ML model confidence: {type_prob:.2f}'
                    }
            
            except Exception as e:
                ml_results = {vtype: {'detected': False, 'confidence': 0.0, 'reason': f'ML error: {str(e)}'} 
                             for vtype in ['advertisement', 'irrelevant', 'fake']}
        
        # Combine rule-based, ML, and LLM results
        final_results = {}
        for violation_type in ['advertisement', 'irrelevant', 'fake']:
            rule_detected = rule_results[violation_type]['detected']
            ml_detected = ml_results.get(violation_type, {}).get('detected', False)
            
            # LLM detection
            llm_detected = False
            llm_conf = 0.0
            llm_reason = ""
            
            if llm_results:
                classification = llm_results['classification']
                if violation_type == 'advertisement':
                    llm_detected = classification.get('is_advertisement', False)
                elif violation_type == 'fake':
                    llm_detected = classification.get('is_fake', False)
                elif violation_type == 'irrelevant':
                    llm_detected = classification.get('is_irrelevant', False)
                
                llm_conf = classification.get('confidence', 0.0) if llm_detected else 0.0
                llm_reason = f"LLM analysis: {'; '.join(classification.get('reasoning', []))}"
            
            # Combine confidences (take maximum, but weight LLM higher)
            rule_conf = rule_results[violation_type]['confidence']
            ml_conf = ml_results.get(violation_type, {}).get('confidence', 0.0)
            
            # Weighted combination: LLM gets 50%, rule-based 30%, traditional ML 20%
            if llm_results:
                combined_conf = (llm_conf * 0.5) + (rule_conf * 0.3) + (ml_conf * 0.2)
            else:
                combined_conf = max(rule_conf, ml_conf)
            
            # Final decision (OR logic - any method detecting triggers)
            final_detected = rule_detected or ml_detected or llm_detected
            
            # Create comprehensive reason
            reasons = []
            if rule_detected:
                reasons.append(f"Rule-based: {rule_results[violation_type]['reason']}")
            if ml_detected:
                reasons.append(f"ML: {ml_results[violation_type]['reason']}")
            if llm_detected and llm_reason:
                reasons.append(llm_reason)
            
            final_results[violation_type] = {
                'detected': final_detected,
                'confidence': combined_conf,
                'reason': ' | '.join(reasons) if reasons else 'No violations detected',
                'methods_detected': [m for m, detected in [
                    ('rule', rule_detected), 
                    ('ml', ml_detected), 
                    ('llm', llm_detected)
                ] if detected]
            }
        
        # Overall violation status
        has_violation = any(result['detected'] for result in final_results.values())
        
        # Calculate quality score with gibberish detection and LLM insights
        features = self.processor.extract_features(cleaned_text)
        quality_score = self.processor.calculate_quality_score(features, text)  # Pass original text
        
        # Enhance quality score with LLM analysis
        if llm_results and 'quality' in llm_results:
            llm_quality = llm_results['quality']['overall_quality']
            # Weighted combination: traditional 60%, LLM 40%
            quality_score = (quality_score * 0.6) + (llm_quality * 0.4)
        
        # Check for gibberish/invalid content
        validation = self.processor.validate_review_content(text)
        if not validation['is_valid']:
            # Mark as violation if content is invalid
            has_violation = True
            final_results['gibberish'] = {
                'detected': True,
                'confidence': 1.0 - validation['confidence'],
                'reason': f"Invalid content: {'; '.join(validation['issues'])}"
            }
        
        # User metadata analysis (if provided)
        user_suspicion_analysis = None
        if user_metadata:
            user_suspicion_analysis = self._analyze_user_suspicion(user_metadata, final_results)
            
            # Apply user suspicion to violation detection
            if user_suspicion_analysis['high_risk']:
                # Enhance existing violations or add user-based suspicion
                for violation_type in final_results:
                    if user_suspicion_analysis['enhanced_violations'].get(violation_type, False):
                        final_results[violation_type]['detected'] = True
                        final_results[violation_type]['confidence'] = max(
                            final_results[violation_type]['confidence'], 
                            user_suspicion_analysis['user_confidence']
                        )
                        if 'user' not in final_results[violation_type]['methods_detected']:
                            final_results[violation_type]['methods_detected'].append('user')
                            final_results[violation_type]['reason'] += f" | User metadata: {user_suspicion_analysis['reason']}"
                
                has_violation = True  # Override if user is high risk
                
                # Reduce quality score based on user suspicion
                quality_score *= (1 - user_suspicion_analysis['suspicion_penalty'])
        
        # Adjust quality score based on violations
        if has_violation:
            quality_score *= 0.5  # Reduce quality score for violations
        
        # Overall confidence (average of individual confidences)
        overall_confidence = np.mean([result['confidence'] for result in final_results.values()])
        
        result = {
            'has_violation': has_violation,
            'violations': final_results,
            'quality_score': quality_score,
            'confidence': overall_confidence,
            'explanation': self._generate_explanation(final_results, has_violation, quality_score)
        }
        
        # Add context information if available
        if context_analysis:
            result['context'] = {
                'product_info': product_info,
                'detected_context': context_analysis['context_analysis']['detected_context'],
                'relevance_score': context_analysis['context_analysis']['relevance_score'],
                'context_keywords': context_analysis['context_analysis']['context_keywords'],
                'overall_assessment': context_analysis['overall_assessment']
            }
        
        # Add user suspicion analysis if available
        if user_suspicion_analysis:
            result['user_analysis'] = user_suspicion_analysis
        
        # Add translation information if available (show for non-English languages)
        if translation_result and translation_result.get('source_language', 'en') != 'en':
            result['translation'] = {
                'original_text': translation_result['original_text'],
                'translated_text': translation_result['translated_text'],
                'source_language': translation_result['source_language'],
                'source_language_name': translation_result['source_language_name'],
                'confidence': translation_result['confidence'],
                'translation_needed': translation_result.get('translation_needed', False),
                'analyzed_original': analyze_original  # Track which text was analyzed
            }
        
        return result
    
    def _generate_explanation(self, violations: Dict[str, Any], has_violation: bool, quality_score: float) -> str:
        """Generate human-readable explanation"""
        
        # Check for gibberish first
        if 'gibberish' in violations and violations['gibberish']['detected']:
            return f"INVALID CONTENT: {violations['gibberish']['reason']} This content appears to be invalid or gibberish."
        
        if not has_violation:
            if quality_score > 0.7:
                return "This review appears to be genuine and high-quality with no policy violations detected."
            elif quality_score > 0.5:
                return "No policy violations detected. Review quality is acceptable but could be more detailed."
            else:
                return "No policy violations detected, but review quality is low due to lack of specific details or unclear content."
        
        violation_types = [vtype for vtype, details in violations.items() if details['detected']]
        
        if len(violation_types) == 1:
            vtype = violation_types[0]
            if vtype == 'gibberish':
                return f"CONTENT VALIDATION FAILED: {violations[vtype]['reason']}"
            else:
                return f"{vtype.title()} violation detected. {violations[vtype]['reason']}"
        elif len(violation_types) > 1:
            return f"Multiple violations detected: {', '.join(violation_types)}. Review should be flagged for manual review."
        else:
            return "Analysis completed with mixed signals. Manual review recommended."
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict policy violations for multiple reviews"""
        results = []
        for text in texts:
            try:
                result = self.predict_single(text)
                results.append(result)
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    'has_violation': False,
                    'violations': {},
                    'quality_score': 0.0,
                    'confidence': 0.0,
                    'explanation': f'Error processing review: {str(e)}'
                })
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models"""
        return {
            'is_trained': self.is_trained,
            'models_available': list(self.models.keys()),
            'vectorizers_available': list(self.vectorizers.keys()),
            'violation_types': ['advertisement', 'irrelevant', 'fake'],
            'approach': 'Ensemble of rule-based and ML classifiers'
        }
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        if self.is_trained:
            model_data = {
                'models': self.models,
                'vectorizers': self.vectorizers,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.vectorizers = model_data['vectorizers']
            self.is_trained = model_data['is_trained']
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def _analyze_user_suspicion(self, user_metadata: Dict[str, Any], existing_violations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user metadata for suspicious patterns and enhance violation detection
        """
        suspicion_indicators = []
        suspicion_score = 0.0
        enhanced_violations = {}
        
        # Check for high suspicion account indicators
        if user_metadata.get('is_high_suspicion_account', False):
            suspicion_indicators.append("Account marked as high suspicion")
            suspicion_score += 0.4
            enhanced_violations['fake'] = True  # High suspicion accounts likely post fake reviews
        
        # Check for burst activity patterns
        if user_metadata.get('has_burst_activity', False):
            suspicion_indicators.append("Burst review activity detected")
            suspicion_score += 0.3
            enhanced_violations['fake'] = True
        
        # Check composite suspicion score
        composite_score = user_metadata.get('composite_suspicion_score', 0.0)
        if composite_score > 0.6:
            suspicion_indicators.append(f"High composite suspicion score ({composite_score:.2f})")
            suspicion_score += min(composite_score, 0.3)  # Cap at 0.3
            enhanced_violations['fake'] = True
        
        # Check account age and review patterns
        account_suspicion = user_metadata.get('account_suspicion_score', 0.0)
        if account_suspicion > 0.5:
            suspicion_indicators.append("Suspicious account characteristics")
            suspicion_score += 0.2
            enhanced_violations['fake'] = True
        
        # Check for promotional patterns (links, domains)
        if user_metadata.get('suspicious_links', False):
            suspicion_indicators.append("History of posting links")
            suspicion_score += 0.2
            enhanced_violations['advertisement'] = True
        
        if user_metadata.get('suspicious_domains', False):
            suspicion_indicators.append("History of suspicious domains")
            suspicion_score += 0.2
            enhanced_violations['advertisement'] = True
        
        # Check device/behavior patterns
        suspicion_level = user_metadata.get('suspicion_level', 'unknown')
        if suspicion_level == 'high':
            suspicion_indicators.append("Behavioral analysis indicates high risk")
            suspicion_score += 0.3
            enhanced_violations['fake'] = True
        elif suspicion_level == 'medium':
            suspicion_indicators.append("Behavioral analysis indicates medium risk")
            suspicion_score += 0.15
        
        # Additional metadata checks
        if user_metadata.get('is_deleted', False):
            suspicion_indicators.append("Account was deleted")
            suspicion_score += 0.15
        
        if user_metadata.get('number_of_reviews', 0) > 100:
            suspicion_indicators.append("Extremely high review count")
            suspicion_score += 0.1
        
        # Calculate final risk assessment
        suspicion_score = min(suspicion_score, 1.0)  # Cap at 1.0
        high_risk = suspicion_score > 0.5
        
        # Calculate penalty to apply to quality score
        suspicion_penalty = suspicion_score * 0.3  # Up to 30% quality penalty
        
        return {
            'high_risk': high_risk,
            'suspicion_score': suspicion_score,
            'suspicion_indicators': suspicion_indicators,
            'enhanced_violations': enhanced_violations,
            'user_confidence': min(0.8, suspicion_score + 0.2),  # Confidence for user-based detection
            'suspicion_penalty': suspicion_penalty,
            'reason': '; '.join(suspicion_indicators) if suspicion_indicators else "No suspicious patterns detected",
            'recommendation': self._get_user_recommendation(suspicion_score, suspicion_indicators)
        }
    
    def _get_user_recommendation(self, suspicion_score: float, indicators: List[str]) -> str:
        """Generate recommendation based on user suspicion analysis"""
        if suspicion_score > 0.7:
            return "HIGH RISK: Strong recommendation to block/flag this user account"
        elif suspicion_score > 0.5:
            return "MEDIUM RISK: Manual review recommended before accepting reviews"
        elif suspicion_score > 0.3:
            return "LOW-MEDIUM RISK: Monitor user activity closely"
        else:
            return "LOW RISK: User appears legitimate based on available data"
import pandas as pd
import numpy as np
import re
import string
from textblob import TextBlob
from typing import Dict, List, Any, Tuple
import nltk
from collections import Counter

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    STOPWORDS = set(stopwords.words('english'))
    SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
except:
    STOPWORDS = set()
    SENTIMENT_ANALYZER = None

class ReviewProcessor:
    """Advanced review preprocessing and feature extraction"""
    
    def __init__(self):
        self.promotional_keywords = [
            'discount', 'sale', 'offer', 'deal', 'coupon', 'promo', 'free shipping',
            'limited time', 'special offer', 'buy now', 'click here', 'visit our',
            'website', 'link', 'url', 'http', 'www', '.com', 'promotion'
        ]
        
        self.irrelevant_indicators = [
            'off topic', 'unrelated', 'wrong place', 'different location',
            'not about', 'irrelevant', 'random', 'spam'
        ]
        
        self.fake_indicators = [
            'perfect', 'amazing', 'best ever', 'highly recommend', 'must visit',
            'absolutely', 'definitely', 'incredible', 'outstanding', 'phenomenal'
        ]
        
        self.quality_indicators = {
            'specific_details': ['time', 'date', 'price', 'location', 'staff', 'menu', 'atmosphere'],
            'balanced_opinion': ['however', 'but', 'although', 'except', 'despite'],
            'personal_experience': ['i', 'my', 'we', 'our', 'personally', 'visited', 'went']
        }
        
        # Common English words for language validation
        self.common_english_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'this', 'that', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my',
            'your', 'his', 'her', 'its', 'our', 'their', 'good', 'great', 'bad', 'nice', 'food',
            'service', 'restaurant', 'place', 'time', 'very', 'really', 'quite', 'well', 'staff',
            'delicious', 'tasty', 'menu', 'order', 'meal', 'lunch', 'dinner', 'breakfast'
        }
        
        # Common Chinese words and characters for validation
        self.common_chinese_chars = {
            '的', '了', '是', '在', '我', '有', '他', '这', '为', '之', '大', '来', '以', '个', '中', '上', 
            '们', '到', '说', '国', '和', '地', '也', '子', '时', '道', '出', '而', '要', '于', '就', '下',
            '得', '可', '你', '年', '生', '自', '会', '那', '后', '能', '对', '着', '事', '她', '里', '所',
            '去', '行', '过', '家', '十', '用', '发', '天', '如', '然', '作', '方', '成', '者', '多', '日',
            '都', '三', '小', '军', '二', '无', '同', '么', '经', '法', '当', '起', '与', '好', '看', '学',
            '进', '种', '将', '还', '分', '此', '心', '前', '面', '又', '定', '见', '只', '主', '没', '公',
            '从', '第', '位', '长', '通', '把', '或', '给', '因', '由', '手', '体', '系', '知', '水', '先',
            '名', '点', '力', '明', '高', '次', '实', '现', '情', '理', '动', '相', '真', '全', '二', '新',
            '已', '工', '被', '门', '等', '战', '很', '最', '关', '开', '本', '走', '回', '间', '业', '使',
            '想', '数', '产', '表', '民', '义', '向', '道', '重', '题', '党', '打', '比', '变', '统', '便',
            '维', '风', '女', '展', '科', '快', '却', '先', '口', '由', '死', '安', '写', '性', '马', '代',
            '感', '级', '构', '物', '立', '4', '取', '平', '住', '异', '外', '创', '钟', '汽', '飞', '它',
            '由', '增', '意', '教', '化', '较', '切', '常', '强', '极', '德', '交', '约', '式', '转', '济',
            '把', '界', '备', '万', '集', '每', '务', '合', '员', '电', '非', '花', '正', '受', '思', '响',
            '白', '美', '价', '组', '书', '活', '须', '众', '音', '影', '歌', '完', '建', '站', '农', '欢',
            '买', '世', '专', '判', '育', '该', '装', '语', '更', '失', '士', '供', '何', '评', '补', '命',
            '识', '友', '联', '网', '收', '候', '产', '品', '企', '资', '总', '广', '件', '术', '存', '原',
            '急', '简', '毎', '客', '报', '博', '精', '际', '程', '护', '病', '史', '英', '永', '师', '市',
            '沈', '余', '论', '试', '告', '类', '海', '路', '今', '消', '诉', '材', '针', '怎', '基', '青',
            # Food/restaurant related Chinese words
            '餐厅', '饭店', '美食', '好吃', '味道', '服务', '环境', '菜单', '价格', '推荐', '食物', '菜品',
            '口味', '新鲜', '干净', '氛围', '员工', '服务员', '厨师', '位置', '停车', '预约', '等待', '排队',
            '点菜', '结账', '满意', '失望', '一般', '不错', '很棒', '非常', '特别', '最好', '最差', '值得',
            '再来', '回去', '朋友', '家人', '聚餐', '约会', '生日', '庆祝', '节日', '周末', '晚餐', '午餐',
            '早餐', '夜宵', '饮料', '茶水', '啤酒', '红酒', '果汁', '汤', '面条', '米饭', '蔬菜', '肉类'
        }
        
        # Language pattern detection
        self.language_patterns = {
            'chinese': r'[\u4e00-\u9fff]',  # Chinese characters
            'japanese': r'[\u3040-\u309f\u30a0-\u30ff]',  # Hiragana/Katakana
            'korean': r'[\uac00-\ud7af]',  # Korean
            'arabic': r'[\u0600-\u06ff]',  # Arabic
            'hindi': r'[\u0900-\u097f]',  # Hindi/Devanagari
            'thai': r'[\u0e00-\u0e7f]',  # Thai
            'cyrillic': r'[\u0400-\u04ff]'  # Russian/Cyrillic
        }
        
        # Restaurant/review related words
        self.review_context_words = {
            'food', 'service', 'restaurant', 'staff', 'menu', 'meal', 'order', 'waiter', 'waitress',
            'chef', 'cook', 'kitchen', 'table', 'seat', 'atmosphere', 'ambiance', 'price', 'cost',
            'bill', 'tip', 'delicious', 'tasty', 'flavor', 'taste', 'fresh', 'hot', 'cold', 'warm',
            'lunch', 'dinner', 'breakfast', 'brunch', 'drink', 'beverage', 'wine', 'beer', 'coffee',
            'tea', 'dessert', 'appetizer', 'main', 'course', 'portion', 'size', 'quality', 'experience'
        }
        
        # Product context categories and their relevant terms
        self.product_contexts = {
            'restaurant': {
                'keywords': ['restaurant', 'cafe', 'diner', 'bistro', 'eatery', 'bar', 'grill', 'kitchen'],
                'relevant_terms': ['food', 'service', 'menu', 'staff', 'atmosphere', 'meal', 'dining'],
                'policy_focus': ['fake_enthusiasm', 'competitor_mentions', 'health_violations']
            },
            'hotel': {
                'keywords': ['hotel', 'motel', 'inn', 'resort', 'lodge', 'hostel', 'accommodation'],
                'relevant_terms': ['room', 'bed', 'bathroom', 'amenities', 'location', 'staff', 'service'],
                'policy_focus': ['fake_bookings', 'competitor_hotels', 'safety_issues']
            },
            'product': {
                'keywords': ['product', 'item', 'device', 'gadget', 'tool', 'equipment'],
                'relevant_terms': ['quality', 'price', 'shipping', 'packaging', 'warranty', 'features'],
                'policy_focus': ['fake_purchases', 'competitor_products', 'safety_concerns']
            },
            'service': {
                'keywords': ['service', 'company', 'business', 'provider', 'agency'],
                'relevant_terms': ['customer service', 'support', 'experience', 'staff', 'process'],
                'policy_focus': ['fake_testimonials', 'competitor_services', 'misleading_claims']
            }
        }
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        return text.strip()
    
    def detect_language(self, text: str) -> str:
        """Detect the primary language of the text"""
        if not text or len(text.strip()) < 3:
            return 'unknown'
        
        # Check for non-Latin scripts first
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, text):
                return lang
        
        # Default to English for Latin scripts
        return 'english'
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive features from review text"""
        features = {}
        
        # Basic text features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character analysis
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['punctuation_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
        
        # Sentiment analysis
        if SENTIMENT_ANALYZER:
            sentiment_scores = SENTIMENT_ANALYZER.polarity_scores(text)
            features.update({
                'sentiment_positive': sentiment_scores['pos'],
                'sentiment_negative': sentiment_scores['neg'],
                'sentiment_neutral': sentiment_scores['neu'],
                'sentiment_compound': sentiment_scores['compound']
            })
        else:
            # Fallback using TextBlob
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        
        # Policy violation indicators
        features['promotional_keywords'] = self._count_keywords(text, self.promotional_keywords)
        features['irrelevant_indicators'] = self._count_keywords(text, self.irrelevant_indicators)
        features['fake_indicators'] = self._count_keywords(text, self.fake_indicators)
        
        # Quality indicators
        features['specific_details'] = self._count_keywords(text, self.quality_indicators['specific_details'])
        features['balanced_opinion'] = self._count_keywords(text, self.quality_indicators['balanced_opinion'])
        features['personal_experience'] = self._count_keywords(text, self.quality_indicators['personal_experience'])
        
        # Advanced linguistic features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capital_words'] = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
        
        # Repetition analysis
        words = text.lower().split()
        word_freq = Counter(words)
        features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
        features['max_word_freq'] = max(word_freq.values()) if word_freq else 0
        
        return features
    
    def _count_keywords(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keywords in text"""
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)
    
    def detect_gibberish(self, text: str) -> Dict[str, Any]:
        """Detect if text is gibberish or nonsensical - now language-aware"""
        if not text or len(text.strip()) < 3:
            return {'is_gibberish': True, 'reason': 'Text too short', 'confidence': 1.0}
        
        # Detect the language first
        detected_lang = self.detect_language(text)
        
        # Handle different languages
        if detected_lang == 'chinese':
            return self._detect_chinese_gibberish(text)
        elif detected_lang in ['japanese', 'korean', 'arabic', 'hindi', 'thai', 'cyrillic']:
            return self._detect_non_latin_gibberish(text, detected_lang)
        else:
            return self._detect_english_gibberish(text)
    
    def _detect_chinese_gibberish(self, text: str) -> Dict[str, Any]:
        """Detect gibberish in Chinese text"""
        gibberish_indicators = 0
        total_checks = 5
        reasons = []
        
        # Remove punctuation for analysis
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        if not chinese_chars:
            return {'is_gibberish': True, 'reason': 'No Chinese characters found', 'confidence': 1.0}
        
        # Check 1: Ratio of common Chinese characters
        common_char_count = sum(1 for char in chinese_chars if char in self.common_chinese_chars)
        if chinese_chars:
            common_char_ratio = common_char_count / len(chinese_chars)
            if common_char_ratio < 0.3:  # Less than 30% common characters
                gibberish_indicators += 1
                reasons.append(f"Few common Chinese characters ({common_char_ratio:.1%})")
        
        # Check 2: Excessive repetition of same character
        char_freq = Counter(chinese_chars)
        if char_freq:
            max_freq = max(char_freq.values())
            if max_freq > len(chinese_chars) * 0.4:  # More than 40% same character
                gibberish_indicators += 1
                most_repeated = max(char_freq.keys(), key=lambda x: char_freq[x])
                reasons.append(f"Excessive repetition of '{most_repeated}'")
        
        # Check 3: Check for keyboard mashing patterns (adjacent keys)
        keyboard_patterns = ['qwer', 'asdf', 'zxcv', '1234', '!@#$']
        text_lower = text.lower()
        for pattern in keyboard_patterns:
            if pattern in text_lower:
                gibberish_indicators += 1
                reasons.append(f"Keyboard mashing pattern: {pattern}")
                break
        
        # Check 4: Very short but claims to be meaningful
        if len(chinese_chars) < 2:
            gibberish_indicators += 1
            reasons.append("Too few Chinese characters")
        
        # Check 5: Only punctuation or numbers
        text_clean = re.sub(r'[\u4e00-\u9fff\s]', '', text)
        if len(text_clean) > len(text) * 0.8:  # More than 80% non-Chinese
            gibberish_indicators += 1
            reasons.append("Mostly punctuation or numbers")
        
        confidence = gibberish_indicators / total_checks
        is_gibberish = confidence > 0.6  # Slightly more lenient for Chinese
        
        return {
            'is_gibberish': is_gibberish,
            'confidence': confidence,
            'reason': '; '.join(reasons) if reasons else 'Chinese text appears valid',
            'failed_checks': gibberish_indicators,
            'total_checks': total_checks,
            'language': 'chinese'
        }
    
    def _detect_non_latin_gibberish(self, text: str, language: str) -> Dict[str, Any]:
        """Detect gibberish in non-Latin scripts (basic check)"""
        gibberish_indicators = 0
        total_checks = 3
        reasons = []
        
        # Check 1: Excessive repetition
        repeated_chars = re.findall(r'(.)\1{4,}', text)
        if repeated_chars:
            gibberish_indicators += 1
            reasons.append("Excessive character repetition")
        
        # Check 2: Very short content
        script_chars = len(re.findall(self.language_patterns[language], text))
        if script_chars < 3:
            gibberish_indicators += 1
            reasons.append("Too few characters in detected script")
        
        # Check 3: Only punctuation
        if len(re.sub(r'[^\w\s]', '', text).strip()) == 0:
            gibberish_indicators += 1
            reasons.append("Only punctuation")
        
        confidence = gibberish_indicators / total_checks
        is_gibberish = confidence > 0.5
        
        return {
            'is_gibberish': is_gibberish,
            'confidence': confidence,
            'reason': '; '.join(reasons) if reasons else f'{language.title()} text appears valid',
            'failed_checks': gibberish_indicators,
            'total_checks': total_checks,
            'language': language
        }
    
    def _detect_english_gibberish(self, text: str) -> Dict[str, Any]:
        """Detect gibberish in English text (original logic)"""
        words = text.lower().split()
        if not words:
            return {'is_gibberish': True, 'reason': 'No words found', 'confidence': 1.0}
        
        gibberish_indicators = 0
        total_checks = 6
        reasons = []
        
        # Check 1: Ratio of common English words
        common_word_count = sum(1 for word in words if word in self.common_english_words)
        common_word_ratio = common_word_count / len(words)
        if common_word_ratio < 0.2:  # Less than 20% common words
            gibberish_indicators += 1
            reasons.append(f"Few recognizable words ({common_word_ratio:.1%})")
        
        # Check 2: Average word length (gibberish tends to have unusual lengths)
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length > 12 or avg_word_length < 2:
            gibberish_indicators += 1
            reasons.append(f"Unusual word length (avg: {avg_word_length:.1f})")
        
        # Check 3: Vowel ratio (real words have reasonable vowel ratios)
        text_clean = re.sub(r'[^a-zA-Z]', '', text.lower())
        if text_clean:
            vowel_count = sum(1 for char in text_clean if char in 'aeiou')
            vowel_ratio = vowel_count / len(text_clean)
            if vowel_ratio < 0.15 or vowel_ratio > 0.6:  # Too few or too many vowels
                gibberish_indicators += 1
                reasons.append(f"Unusual vowel ratio ({vowel_ratio:.1%})")
        
        # Check 4: Consonant clusters (gibberish often has long consonant sequences)
        consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyz]{4,}', text.lower())
        if len(consonant_clusters) > 0:
            gibberish_indicators += 1
            reasons.append(f"Long consonant sequences: {consonant_clusters}")
        
        # Check 5: Repeating character patterns
        repeated_chars = re.findall(r'(.)\1{3,}', text.lower())
        if repeated_chars:
            gibberish_indicators += 1
            reasons.append(f"Excessive character repetition: {''.join(repeated_chars)}")
        
        # Check 6: Contains any review-relevant words
        review_word_count = sum(1 for word in words if word in self.review_context_words)
        if len(words) > 3 and review_word_count == 0:
            gibberish_indicators += 1
            reasons.append("No restaurant/review related words")
        
        # Calculate confidence based on number of failed checks
        confidence = gibberish_indicators / total_checks
        is_gibberish = confidence > 0.5  # More than half the checks failed
        
        return {
            'is_gibberish': is_gibberish,
            'confidence': confidence,
            'reason': '; '.join(reasons) if reasons else 'Text appears valid',
            'failed_checks': gibberish_indicators,
            'total_checks': total_checks,
            'language': 'english'
        }
    
    def validate_review_content(self, text: str) -> Dict[str, Any]:
        """Validate if text looks like a legitimate review - language-aware"""
        validation = {
            'is_valid': True,
            'issues': [],
            'confidence': 1.0
        }
        
        # Check for gibberish first
        gibberish_result = self.detect_gibberish(text)
        if gibberish_result['is_gibberish']:
            validation['is_valid'] = False
            validation['issues'].append(f"Gibberish detected: {gibberish_result['reason']}")
            validation['confidence'] = 1 - gibberish_result['confidence']
        
        # Language-specific length validation
        detected_lang = self.detect_language(text)
        
        if detected_lang == 'chinese':
            # Chinese text is more compact - count Chinese characters instead of total length
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            if chinese_chars < 3:  # At least 3 Chinese characters
                validation['is_valid'] = False
                validation['issues'].append("Review too short to be meaningful (less than 3 Chinese characters)")
        elif detected_lang in ['japanese', 'korean', 'arabic', 'hindi', 'thai', 'cyrillic']:
            # Other non-Latin scripts - use character count
            script_chars = len(re.findall(self.language_patterns[detected_lang], text))
            if script_chars < 4:
                validation['is_valid'] = False
                validation['issues'].append(f"Review too short to be meaningful (less than 4 {detected_lang} characters)")
        else:
            # English and Latin scripts - use word count
            words = text.split()
            if len(words) < 3:  # At least 3 words
                validation['is_valid'] = False
                validation['issues'].append("Review too short to be meaningful (less than 3 words)")
        
        # Check for excessive special characters (but be more lenient)
        special_char_ratio = sum(1 for char in text if char in '!@#$%^&*(){}[]|\\:";\'<>?,./`~') / len(text) if text else 0
        if special_char_ratio > 0.5:  # Increased from 0.3 to 0.5
            validation['is_valid'] = False
            validation['issues'].append(f"Excessive special characters ({special_char_ratio:.1%})")
        
        # Check for reasonable sentence structure (skip for Chinese)
        if detected_lang not in ['chinese', 'japanese', 'korean']:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            words = text.split()
            if len(sentences) == 0 and len(words) > 10:
                validation['issues'].append("No clear sentence structure")
        
        return validation
    
    def detect_product_context(self, text: str, product_info: str = None) -> Dict[str, Any]:
        """Detect and analyze product context from review text and product information"""
        context_analysis = {
            'detected_context': 'general',
            'confidence': 0.5,
            'relevance_score': 0.5,
            'context_keywords': [],
            'product_match': False,
            'product_info': product_info
        }
        
        text_lower = text.lower()
        
        # If product info is provided, use it for context
        if product_info:
            product_info_lower = product_info.lower()
            
            # Check which product context best matches
            best_match = 'general'
            best_score = 0
            
            for context_type, context_data in self.product_contexts.items():
                score = 0
                matched_keywords = []
                
                # Check product info for context keywords
                for keyword in context_data['keywords']:
                    if keyword in product_info_lower:
                        score += 0.3
                        matched_keywords.append(keyword)
                
                # Check review text for relevant terms
                for term in context_data['relevant_terms']:
                    if term in text_lower:
                        score += 0.1
                        matched_keywords.append(term)
                
                if score > best_score:
                    best_score = score
                    best_match = context_type
                    context_analysis['context_keywords'] = matched_keywords
            
            context_analysis['detected_context'] = best_match
            context_analysis['confidence'] = min(1.0, best_score)
            context_analysis['product_match'] = best_score > 0.3
        
        else:
            # Auto-detect context from review text only
            for context_type, context_data in self.product_contexts.items():
                score = 0
                matched_keywords = []
                
                for keyword in context_data['keywords']:
                    if keyword in text_lower:
                        score += 0.2
                        matched_keywords.append(keyword)
                
                for term in context_data['relevant_terms']:
                    if term in text_lower:
                        score += 0.1
                        matched_keywords.append(term)
                
                if score > context_analysis['confidence']:
                    context_analysis['detected_context'] = context_type
                    context_analysis['confidence'] = min(1.0, score)
                    context_analysis['context_keywords'] = matched_keywords
        
        # Calculate relevance score based on context match
        context_analysis['relevance_score'] = self._calculate_context_relevance(
            text, context_analysis['detected_context'], product_info
        )
        
        return context_analysis
    
    def _calculate_context_relevance(self, text: str, context_type: str, product_info: str = None) -> float:
        """Calculate how relevant the review is to the product context"""
        if context_type == 'general':
            return 0.5  # Neutral relevance for general context
        
        context_data = self.product_contexts.get(context_type, {})
        relevant_terms = context_data.get('relevant_terms', [])
        keywords = context_data.get('keywords', [])
        
        text_lower = text.lower()
        words = text.split()
        
        # Start with very low relevance - must prove relevance
        relevance_score = 0.0
        
        # Count direct context keyword matches (strong signal)
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        if keyword_matches > 0:
            relevance_score += keyword_matches * 0.3
        
        # Count relevant terms (weaker signal)
        relevant_term_count = sum(1 for term in relevant_terms if term in text_lower)
        if relevant_term_count > 0:
            relevance_score += relevant_term_count * 0.15
        
        # Require minimum threshold of context-relevant content
        # If no context keywords OR relevant terms, severely penalize
        if keyword_matches == 0 and relevant_term_count < 2:
            relevance_score = max(0.1, relevance_score * 0.2)  # Very low relevance
        
        # Additional context-specific checks
        if context_type == 'restaurant':
            # Look for food/dining related words
            food_words = ['food', 'eat', 'meal', 'lunch', 'dinner', 'breakfast', 'taste', 'flavor', 
                         'delicious', 'cook', 'order', 'menu', 'waiter', 'server', 'dish', 'cuisine']
            food_matches = sum(1 for word in food_words if word in text_lower)
            if food_matches > 0:
                relevance_score += food_matches * 0.1
            elif len(words) > 5:  # For longer texts, expect some food context
                relevance_score *= 0.5
                
        elif context_type == 'hotel':
            # Look for accommodation related words
            hotel_words = ['stay', 'room', 'bed', 'sleep', 'night', 'check', 'lobby', 'front desk',
                          'bathroom', 'shower', 'towel', 'clean', 'booking', 'reservation']
            hotel_matches = sum(1 for word in hotel_words if word in text_lower)
            if hotel_matches > 0:
                relevance_score += hotel_matches * 0.1
            elif len(words) > 5:
                relevance_score *= 0.5
                
        elif context_type == 'product':
            # Look for product/purchase related words
            product_words = ['buy', 'purchase', 'order', 'delivery', 'quality', 'price', 'value',
                           'feature', 'work', 'use', 'recommend', 'product', 'item']
            product_matches = sum(1 for word in product_words if word in text_lower)
            if product_matches > 0:
                relevance_score += product_matches * 0.1
            elif len(words) > 5:
                relevance_score *= 0.5
        
        # Check for completely irrelevant topics
        irrelevant_topics = self._detect_irrelevant_topics(text_lower, context_type)
        if irrelevant_topics:
            relevance_score *= 0.1  # Severely penalize irrelevant topics
        
        # Boost if product info strongly matches
        if product_info:
            product_keywords = context_data.get('keywords', [])
            product_matches = sum(1 for keyword in product_keywords 
                                if keyword in product_info.lower())
            if product_matches > 0:
                relevance_score += 0.15
        
        return min(1.0, max(0.0, relevance_score))
    
    def _detect_irrelevant_topics(self, text_lower: str, context_type: str) -> List[str]:
        """Detect if text contains topics completely irrelevant to context"""
        irrelevant_topics = []
        
        # Define irrelevant topic patterns for each context
        irrelevant_patterns = {
            'restaurant': {
                'celebrities': ['taylor swift', 'celebrity', 'famous', 'singer', 'actor', 'actress', 'movie star'],
                'technology': ['computer', 'software', 'app', 'download', 'internet', 'wifi', 'bluetooth'],
                'sports': ['football', 'basketball', 'soccer', 'game', 'sport', 'team', 'player'],
                'politics': ['president', 'election', 'vote', 'government', 'political', 'congress'],
                'personal_life': ['my dog', 'my cat', 'my family', 'my job', 'my house', 'my car']
            },
            'hotel': {
                'celebrities': ['taylor swift', 'celebrity', 'famous', 'singer', 'actor', 'actress'],
                'food_focus': ['delicious', 'tasty', 'flavor', 'recipe', 'cooking', 'chef'],
                'sports': ['football', 'basketball', 'soccer', 'game', 'sport', 'team'],
                'politics': ['president', 'election', 'vote', 'government', 'political']
            },
            'product': {
                'celebrities': ['taylor swift', 'celebrity', 'famous', 'singer', 'actor'],
                'food': ['delicious', 'tasty', 'meal', 'restaurant', 'dinner', 'lunch'],
                'accommodation': ['hotel', 'room', 'bed', 'stay', 'check in'],
                'politics': ['president', 'election', 'vote', 'government']
            },
            'service': {
                'celebrities': ['taylor swift', 'celebrity', 'famous', 'singer', 'actor'],
                'food': ['delicious', 'tasty', 'meal', 'restaurant', 'dinner'],
                'products': ['buy', 'purchase', 'item', 'shipping', 'delivery'],
                'politics': ['president', 'election', 'vote', 'government']
            }
        }
        
        context_irrelevant = irrelevant_patterns.get(context_type, {})
        
        for topic_name, topic_words in context_irrelevant.items():
            for word_phrase in topic_words:
                if word_phrase in text_lower:
                    irrelevant_topics.append(f"{topic_name}:{word_phrase}")
        
        return irrelevant_topics
    
    def analyze_with_context(self, text: str, product_info: str = None) -> Dict[str, Any]:
        """Comprehensive analysis including product context"""
        # Get basic features
        features = self.extract_features(text)
        
        # Get context analysis
        context_analysis = self.detect_product_context(text, product_info)
        
        # Enhanced quality score with context
        quality_score = self.calculate_quality_score(features, text)
        
        # Adjust quality based on context relevance (much stricter penalties)
        context_relevance = context_analysis['relevance_score']
        if context_relevance < 0.2:
            quality_score *= 0.1  # Severe penalty for very irrelevant content
        elif context_relevance < 0.4:
            quality_score *= 0.3  # Strong penalty for irrelevant content
        elif context_relevance < 0.6:
            quality_score *= 0.7  # Moderate penalty for low relevance
        elif context_relevance > 0.8:
            quality_score *= 1.1  # Boost highly relevant content
            quality_score = min(1.0, quality_score)
        
        # Detect context-specific policy violations
        context_violations = self._detect_context_violations(text, context_analysis, product_info)
        
        return {
            'text': text,
            'product_info': product_info,
            'context_analysis': context_analysis,
            'features': features,
            'quality_score': quality_score,
            'context_violations': context_violations,
            'overall_assessment': self._generate_context_assessment(
                context_analysis, quality_score, context_violations
            )
        }
    
    def _detect_context_violations(self, text: str, context_analysis: Dict, product_info: str = None) -> Dict[str, Any]:
        """Detect policy violations specific to product context"""
        violations = {
            'irrelevant_content': False,
            'competitor_mention': False,
            'context_mismatch': False,
            'details': []
        }
        
        text_lower = text.lower()
        context_type = context_analysis['detected_context']
        relevance_score = context_analysis['relevance_score']
        
        # Check for irrelevant content (much stricter threshold)
        if relevance_score < 0.4:  # Increased from 0.3 to 0.4
            violations['irrelevant_content'] = True
            violations['details'].append(f"Review not relevant to {context_type} context (relevance: {relevance_score:.2f})")
            
        # Additional strict check for very low relevance
        if relevance_score < 0.2:
            violations['irrelevant_content'] = True
            violations['details'].append(f"Very low relevance to {context_type} - likely off-topic content")
        
        # Check for context mismatch
        if product_info and not context_analysis['product_match']:
            # Look for mentions of different product types
            other_contexts = [ctx for ctx in self.product_contexts.keys() if ctx != context_type]
            for other_ctx in other_contexts:
                other_keywords = self.product_contexts[other_ctx]['keywords']
                if any(keyword in text_lower for keyword in other_keywords):
                    violations['context_mismatch'] = True
                    violations['details'].append(f"Review mentions {other_ctx} terms but product is {context_type}")
                    break
        
        # Check for competitor mentions (generic patterns)
        competitor_patterns = [
            r'\b(better than|compared to|versus|vs)\s+[\w\s]+\b',
            r'\b(other|another|different)\s+(restaurant|hotel|product|service)\b',
            r'\b(competitor|rival|alternative)\b'
        ]
        
        for pattern in competitor_patterns:
            if re.search(pattern, text_lower):
                violations['competitor_mention'] = True
                violations['details'].append("Potential competitor comparison detected")
                break
        
        return violations
    
    def _generate_context_assessment(self, context_analysis: Dict, quality_score: float, violations: Dict) -> str:
        """Generate human-readable assessment including context"""
        context_type = context_analysis['detected_context']
        relevance = context_analysis['relevance_score']
        
        assessment_parts = []
        
        # Context relevance
        if relevance > 0.7:
            assessment_parts.append(f"Highly relevant to {context_type}")
        elif relevance > 0.4:
            assessment_parts.append(f"Moderately relevant to {context_type}")
        else:
            assessment_parts.append(f"Low relevance to {context_type}")
        
        # Quality assessment
        if quality_score > 0.7:
            assessment_parts.append("High quality content")
        elif quality_score > 0.4:
            assessment_parts.append("Acceptable quality")
        else:
            assessment_parts.append("Low quality content")
        
        # Violations
        violation_count = sum(1 for v in violations.values() if isinstance(v, bool) and v)
        if violation_count > 0:
            assessment_parts.append(f"Policy concerns: {', '.join(violations['details'])}")
        
        return " | ".join(assessment_parts)
    
    def detect_language_patterns(self, text: str) -> Dict[str, Any]:
        """Detect various language patterns that might indicate issues"""
        patterns = {}
        
        # Repetitive patterns
        words = text.lower().split()
        if words:
            word_freq = Counter(words)
            most_common_word, max_freq = word_freq.most_common(1)[0]
            patterns['repetitive_word'] = max_freq > len(words) * 0.3
            patterns['most_repeated_word'] = most_common_word if patterns['repetitive_word'] else None
        
        # Template-like patterns
        template_phrases = [
            'highly recommend', 'must visit', 'amazing experience',
            'best place', 'definitely worth', 'absolutely loved'
        ]
        patterns['template_phrases'] = sum(1 for phrase in template_phrases if phrase in text.lower())
        
        # Promotional language
        promotional_patterns = [
            r'\b\d+%\s*off\b', r'\bfree\b.*\bshipping\b', r'\bdiscount\b',
            r'\bsale\b', r'\boffer\b', r'\bdeal\b'
        ]
        patterns['promotional_language'] = sum(1 for pattern in promotional_patterns 
                                             if re.search(pattern, text.lower()))
        
        return patterns
    
    def calculate_quality_score(self, features: Dict[str, Any], text: str = "") -> float:
        """Calculate overall review quality score based on features"""
        quality_score = 0.5  # Base score
        
        # First, check for gibberish - this is critical
        if text:
            gibberish_result = self.detect_gibberish(text)
            if gibberish_result['is_gibberish']:
                # Severely penalize gibberish - should be very low quality
                quality_score = 0.1 * (1 - gibberish_result['confidence'])
                return max(0.0, min(1.0, quality_score))
            
            # Also run content validation
            validation = self.validate_review_content(text)
            if not validation['is_valid']:
                quality_score *= validation['confidence']  # Reduce by validation confidence
        
        # Length indicators (optimal length is important)
        word_count = features.get('word_count', 0)
        if 10 <= word_count <= 200:
            quality_score += 0.2
        elif word_count > 200:
            quality_score += 0.1
        elif word_count < 5:  # Very short reviews are low quality
            quality_score -= 0.3
        
        # Specific details boost quality
        if features.get('specific_details', 0) > 0:
            quality_score += 0.15
        
        # Personal experience indicators
        if features.get('personal_experience', 0) > 0:
            quality_score += 0.1
        
        # Balanced opinion (not overly positive/negative)
        if features.get('balanced_opinion', 0) > 0:
            quality_score += 0.1
        
        # Penalize excessive promotional content
        if features.get('promotional_keywords', 0) > 2:
            quality_score -= 0.3
        
        # Penalize excessive fake indicators
        if features.get('fake_indicators', 0) > 3:
            quality_score -= 0.2
        
        # Penalize poor language quality
        if features.get('uppercase_ratio', 0) > 0.3:  # Too much shouting
            quality_score -= 0.1
        
        if features.get('repetitive_word', False):
            quality_score -= 0.15
        
        # Penalize very low unique word ratio (repetitive content)
        unique_ratio = features.get('unique_word_ratio', 1.0)
        if unique_ratio < 0.5:
            quality_score -= 0.2
        
        # Reward reasonable vowel/consonant distribution
        vowel_ratio = features.get('vowel_ratio', 0.3)  # Add this to extract_features if not present
        if 0.2 <= vowel_ratio <= 0.5:
            quality_score += 0.05
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, quality_score))
    
    def process_batch(self, reviews: List[str]) -> pd.DataFrame:
        """Process a batch of reviews and return feature dataframe"""
        processed_data = []
        
        for i, review in enumerate(reviews):
            try:
                cleaned_text = self.preprocess_text(review)
                features = self.extract_features(cleaned_text)
                language_patterns = self.detect_language_patterns(cleaned_text)
                quality_score = self.calculate_quality_score(features, review)  # Pass original text for validation
                
                row_data = {
                    'original_text': review,
                    'cleaned_text': cleaned_text,
                    'quality_score': quality_score,
                    **features,
                    **language_patterns
                }
                
                processed_data.append(row_data)
                
            except Exception as e:
                # Handle errors gracefully
                processed_data.append({
                    'original_text': review,
                    'cleaned_text': '',
                    'quality_score': 0.0,
                    'error': str(e)
                })
        
        return pd.DataFrame(processed_data)
    
    def get_feature_importance(self) -> Dict[str, str]:
        """Return explanation of feature importance for interpretability"""
        return {
            'length': 'Review length in characters',
            'word_count': 'Number of words in review',
            'sentiment_compound': 'Overall sentiment score (-1 to 1)',
            'promotional_keywords': 'Count of promotional terms',
            'specific_details': 'Mentions of specific details',
            'personal_experience': 'Personal experience indicators',
            'quality_score': 'Overall calculated quality score',
            'repetitive_word': 'Whether review has repetitive words',
            'promotional_language': 'Promotional language patterns detected'
        }
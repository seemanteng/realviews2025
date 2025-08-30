"""
Chinese ASAP Dataset Loader for RealViews
Handles the NAACL 2021 ASAP Chinese restaurant review dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASAPChineseLoader:
    """Loader for ASAP Chinese restaurant review dataset"""
    
    def __init__(self):
        # ASAP aspect categories mapping
        self.aspect_categories = {
            'Location#Transportation': 'location_transportation',
            'Location#Downtown': 'location_downtown', 
            'Location#Easy_to_find': 'location_easy_to_find',
            'Service#Queue': 'service_queue',
            'Service#Hospitality': 'service_hospitality',
            'Service#Parking': 'service_parking',
            'Service#Timely': 'service_timely',
            'Price#Level': 'price_level',
            'Price#Cost_effective': 'price_cost_effective',
            'Price#Discount': 'price_discount',
            'Ambience#Decoration': 'ambience_decoration',
            'Ambience#Noise': 'ambience_noise',
            'Ambience#Space': 'ambience_space',
            'Ambience#Sanitary': 'ambience_sanitary',
            'Food#Portion': 'food_portion',
            'Food#Taste': 'food_taste',
            'Food#Appearance': 'food_appearance',
            'Food#Recommend': 'food_recommend'
        }
        
        # Sentiment labels: 1(Positive), 0(Neutral), -1(Negative), -2(Not-Mentioned)
        self.sentiment_labels = {1: 'positive', 0: 'neutral', -1: 'negative', -2: 'not_mentioned'}
    
    def load_asap_dataset(self, file_path: str) -> pd.DataFrame:
        """Load ASAP Chinese dataset from CSV file"""
        try:
            logger.info(f"Loading ASAP dataset from {file_path}")
            
            # Read CSV with proper encoding for Chinese characters
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Remove BOM character if present
            if df.columns[0].startswith('﻿'):
                df.columns = [col.replace('﻿', '') for col in df.columns]
            
            logger.info(f"Loaded {len(df)} Chinese restaurant reviews")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading ASAP dataset: {e}")
            return pd.DataFrame()
    
    def convert_to_realviews_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert ASAP data to RealViews training format"""
        
        training_data = []
        
        for _, row in df.iterrows():
            review_text = row['review']
            star_rating = row['star'] if 'star' in row else None
            
            # Analyze sentiment patterns to infer violation types
            violation_type = self._infer_violation_type(row)
            
            # Create training sample
            sample = {
                'text': review_text,
                'review_text': review_text,  # Alias for compatibility
                'rating': star_rating,
                'language': 'chinese',
                'source': 'asap',
                'violation_type': violation_type,
                'label': violation_type,  # Alias for compatibility
                'quality_score': self._calculate_quality_score(row, star_rating),
                
                # Keep aspect sentiment information for advanced training
                'aspect_sentiments': self._extract_aspect_sentiments(row)
            }
            
            training_data.append(sample)
        
        result_df = pd.DataFrame(training_data)
        logger.info(f"Converted to {len(result_df)} training samples")
        logger.info(f"Violation distribution: {result_df['violation_type'].value_counts().to_dict()}")
        
        return result_df
    
    def _infer_violation_type(self, row: pd.Series) -> str:
        """Infer violation type based on review characteristics and sentiment patterns"""
        
        review_text = str(row['review']).lower()
        star_rating = row['star'] if 'star' in row else 5.0
        
        # Extract aspect sentiments (excluding id, review, star columns)
        aspect_columns = [col for col in row.index if col not in ['id', 'review', 'star']]
        aspect_sentiments = [row[col] for col in aspect_columns if pd.notna(row[col])]
        
        # Count positive/negative/neutral sentiments
        positive_count = sum(1 for sent in aspect_sentiments if sent == 1)
        negative_count = sum(1 for sent in aspect_sentiments if sent == -1)
        neutral_count = sum(1 for sent in aspect_sentiments if sent == 0)
        mentioned_count = positive_count + negative_count + neutral_count
        
        # Detect promotional content (advertisements)
        ad_patterns = [
            r'优惠|折扣|促销|活动|团购',  # Discount/promotion keywords
            r'电话|联系|微信|QQ|地址',    # Contact information
            r'关注|转发|点赞|分享',      # Social media promotion
            r'免费|送|赠|礼品',          # Free gifts/promotions
        ]
        
        ad_matches = sum(1 for pattern in ad_patterns if re.search(pattern, review_text))
        
        if ad_matches >= 2:
            return 'advertisement'
        
        # Detect fake reviews
        # Very extreme ratings with generic positive language
        if star_rating >= 4.5 and positive_count >= 8 and negative_count == 0:
            fake_patterns = [
                r'非常.*非常.*非常',        # Excessive repetition
                r'很.*很.*很.*很',          # Too many "very"
                r'棒.*棒.*棒',             # Repeated "great"
                r'推荐.*推荐',             # Repeated "recommend"
            ]
            
            fake_matches = sum(1 for pattern in fake_patterns if re.search(pattern, review_text))
            
            if fake_matches >= 1 or len(review_text) < 20:
                return 'fake'
        
        # Very low ratings with all negative aspects might be fake negative reviews
        if star_rating <= 2.0 and negative_count >= 6 and positive_count == 0:
            return 'fake'
        
        # Detect irrelevant content
        # Reviews that mention very few aspects (might be off-topic)
        if mentioned_count <= 2 and len(review_text) < 50:
            return 'irrelevant'
        
        # Check for topic irrelevance (non-restaurant content)
        irrelevant_patterns = [
            r'医院|学校|公司|办公|银行',   # Non-restaurant venues
            r'买|购买|网购|淘宝',        # Shopping context
            r'工作|上班|会议',          # Work context
        ]
        
        irrelevant_matches = sum(1 for pattern in irrelevant_patterns if re.search(pattern, review_text))
        if irrelevant_matches >= 1:
            return 'irrelevant'
        
        # Default to clean/no violation
        return 'none'
    
    def _calculate_quality_score(self, row: pd.Series, star_rating: float) -> float:
        """Calculate quality score based on review characteristics"""
        
        review_text = str(row['review'])
        
        # Base score from star rating consistency
        score = 0.5
        
        # Text length factor (longer reviews tend to be higher quality)
        text_length = len(review_text)
        if text_length > 200:
            score += 0.2
        elif text_length > 100:
            score += 0.1
        elif text_length < 30:
            score -= 0.2
        
        # Aspect mention diversity (more aspects = higher quality)
        aspect_columns = [col for col in row.index if col not in ['id', 'review', 'star']]
        mentioned_aspects = sum(1 for col in aspect_columns if row[col] != -2)  # Not "Not-Mentioned"
        
        if mentioned_aspects >= 8:
            score += 0.2
        elif mentioned_aspects >= 5:
            score += 0.1
        elif mentioned_aspects <= 1:
            score -= 0.2
        
        # Rating-sentiment consistency
        positive_aspects = sum(1 for col in aspect_columns if row[col] == 1)
        negative_aspects = sum(1 for col in aspect_columns if row[col] == -1)
        
        # High rating should have more positive aspects
        if star_rating >= 4.0 and positive_aspects > negative_aspects:
            score += 0.1
        # Low rating should have more negative aspects  
        elif star_rating <= 2.0 and negative_aspects > positive_aspects:
            score += 0.1
        # Inconsistent rating-sentiment
        elif (star_rating >= 4.0 and negative_aspects > positive_aspects) or \
             (star_rating <= 2.0 and positive_aspects > negative_aspects):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _extract_aspect_sentiments(self, row: pd.Series) -> Dict:
        """Extract aspect sentiment information"""
        aspect_sentiments = {}
        
        for col in row.index:
            if col not in ['id', 'review', 'star'] and col in self.aspect_categories:
                sentiment_value = row[col]
                if pd.notna(sentiment_value):
                    aspect_sentiments[self.aspect_categories[col]] = self.sentiment_labels.get(sentiment_value, 'unknown')
        
        return aspect_sentiments
    
    def prepare_training_splits(self, df: pd.DataFrame, 
                              train_ratio: float = 0.7, 
                              val_ratio: float = 0.15, 
                              test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets"""
        
        # Shuffle the data
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        n_samples = len(df_shuffled)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        splits = {
            'train': df_shuffled[:n_train],
            'validation': df_shuffled[n_train:n_train + n_val],
            'test': df_shuffled[n_train + n_val:]
        }
        
        logger.info("Chinese dataset splits:")
        for split_name, split_df in splits.items():
            logger.info(f"  {split_name}: {len(split_df)} samples")
            violation_dist = split_df['violation_type'].value_counts()
            logger.info(f"    Violation distribution: {violation_dist.to_dict()}")
        
        return splits

def load_chinese_training_data(file_path: str = "data/raw/dev.csv") -> Dict[str, pd.DataFrame]:
    """Convenience function to load and prepare Chinese training data"""
    
    loader = ASAPChineseLoader()
    
    # Load raw data
    raw_df = loader.load_asap_dataset(file_path)
    if raw_df.empty:
        logger.error("Failed to load Chinese dataset")
        return {}
    
    # Convert to training format
    training_df = loader.convert_to_realviews_format(raw_df)
    
    # Create splits
    splits = loader.prepare_training_splits(training_df)
    
    return splits
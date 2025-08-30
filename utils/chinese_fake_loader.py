"""
Chinese Fake Review Dataset Loader for RealViews
Handles the new chinese_fake.csv dataset with labeled fake/real Chinese reviews
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseFakeLoader:
    """Loader for Chinese fake review dataset"""
    
    def __init__(self):
        # Label mapping: 1 = fake, 0 = real
        self.label_mapping = {1: 'fake', 0: 'none'}
    
    def load_chinese_fake_dataset(self, file_path: str) -> pd.DataFrame:
        """Load Chinese fake dataset from CSV file"""
        try:
            logger.info(f"Loading Chinese fake dataset from {file_path}")
            
            # The file has each row wrapped in quotes with tab separators inside
            # We need to manually parse it
            data_rows = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Remove surrounding quotes and newline
                        line = line.strip()
                        if line.startswith('"') and line.endswith('"'):
                            line = line[1:-1]  # Remove quotes
                        
                        # Split by tab
                        parts = line.split('\t')
                        
                        if len(parts) >= 2:  # At least label and review_text
                            data_rows.append(parts)
                        else:
                            logger.warning(f"Skipping line {line_num}: insufficient columns")
                            
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
            
            if not data_rows:
                raise ValueError("No valid data rows found")
            
            # Convert to DataFrame
            max_cols = max(len(row) for row in data_rows)
            logger.info(f"Max columns found: {max_cols}")
            
            # Pad shorter rows with None
            padded_rows = []
            for row in data_rows:
                if len(row) < max_cols:
                    row.extend([None] * (max_cols - len(row)))
                padded_rows.append(row)
            
            df = pd.DataFrame(padded_rows)
            
            # Assign column names
            expected_cols = ['label', 'review_text', 'date1', 'date2', 'username', 'score']
            df.columns = expected_cols[:min(len(expected_cols), df.shape[1])]
            
            # Add missing columns if needed
            for col in expected_cols[df.shape[1]:]:
                df[col] = None
            
            logger.info(f"Raw data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Clean up the data
            df = df.dropna(subset=['label', 'review_text'])
            
            # Convert label to int, handling any string values
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
            
            # Convert score to numeric if present
            if 'score' in df.columns and df['score'].notna().any():
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
            
            logger.info(f"Cleaned data shape: {df.shape}")
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading Chinese fake dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def convert_to_realviews_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Chinese fake data to RealViews training format"""
        
        training_data = []
        
        for _, row in df.iterrows():
            review_text = str(row['review_text'])
            label = int(row['label'])
            rating = float(row['score']) if pd.notna(row['score']) else None
            
            # Convert binary label to violation type
            violation_type = self.label_mapping[label]
            
            # Create training sample
            sample = {
                'text': review_text,
                'review_text': review_text,  # Alias for compatibility
                'rating': rating,
                'language': 'chinese',
                'source': 'chinese_fake',
                'violation_type': violation_type,
                'label': violation_type,  # Alias for compatibility
                'quality_score': self._calculate_quality_score(review_text, rating, label),
                'original_label': label,  # Keep original binary label
                'username': row['username'],
                'date': row['date1']
            }
            
            training_data.append(sample)
        
        result_df = pd.DataFrame(training_data)
        logger.info(f"Converted to {len(result_df)} training samples")
        logger.info(f"Violation distribution: {result_df['violation_type'].value_counts().to_dict()}")
        
        return result_df
    
    def _calculate_quality_score(self, review_text: str, rating: float, is_fake: int) -> float:
        """Calculate quality score based on review characteristics"""
        
        # Start with base score
        score = 0.5
        
        # Text length factor (longer reviews tend to be higher quality)
        text_length = len(review_text)
        if text_length > 200:
            score += 0.2
        elif text_length > 100:
            score += 0.1
        elif text_length < 30:
            score -= 0.2
        
        # Fake reviews get lower quality scores
        if is_fake == 1:
            score -= 0.3
        else:
            score += 0.2
        
        # Rating consistency (extreme ratings might be suspicious)
        if rating and rating > 0:
            if rating >= 9:  # Very high ratings might be fake
                score -= 0.1
            elif rating <= 3:  # Very low ratings might also be fake
                score -= 0.1
            elif 6 <= rating <= 8:  # Moderate ratings seem more authentic
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def prepare_training_splits(self, df: pd.DataFrame, 
                              train_ratio: float = 0.7, 
                              val_ratio: float = 0.15, 
                              test_ratio: float = 0.15,
                              stratify: bool = True) -> Dict[str, pd.DataFrame]:
        """Split data into train/validation/test sets with stratification"""
        
        if stratify:
            # Stratified split to maintain label balance
            from sklearn.model_selection import train_test_split
            
            # First split: train vs (val + test)
            train_df, temp_df = train_test_split(
                df, 
                test_size=(val_ratio + test_ratio),
                stratify=df['violation_type'],
                random_state=42
            )
            
            # Second split: val vs test
            val_size = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_size),
                stratify=temp_df['violation_type'],
                random_state=42
            )
            
            splits = {
                'train': train_df.reset_index(drop=True),
                'validation': val_df.reset_index(drop=True),
                'test': test_df.reset_index(drop=True)
            }
        else:
            # Simple random split
            df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            
            n_samples = len(df_shuffled)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            splits = {
                'train': df_shuffled[:n_train],
                'validation': df_shuffled[n_train:n_train + n_val],
                'test': df_shuffled[n_train + n_val:]
            }
        
        logger.info("Chinese fake dataset splits:")
        for split_name, split_df in splits.items():
            logger.info(f"  {split_name}: {len(split_df)} samples")
            violation_dist = split_df['violation_type'].value_counts()
            logger.info(f"    Violation distribution: {violation_dist.to_dict()}")
        
        return splits

def load_chinese_fake_training_data(file_path: str = "data/raw/chinese_fake.csv") -> Dict[str, pd.DataFrame]:
    """Convenience function to load and prepare Chinese fake training data"""
    
    loader = ChineseFakeLoader()
    
    # Load raw data
    raw_df = loader.load_chinese_fake_dataset(file_path)
    if raw_df.empty:
        logger.error("Failed to load Chinese fake dataset")
        return {}
    
    # Convert to training format
    training_df = loader.convert_to_realviews_format(raw_df)
    
    # Create stratified splits
    splits = loader.prepare_training_splits(training_df, stratify=True)
    
    return splits

def analyze_chinese_fake_dataset(file_path: str = "data/raw/chinese_fake.csv"):
    """Analyze the Chinese fake dataset structure and content"""
    
    loader = ChineseFakeLoader()
    df = loader.load_chinese_fake_dataset(file_path)
    
    if df.empty:
        print("Failed to load dataset")
        return
    
    print(f"\n📊 Chinese Fake Dataset Analysis")
    print(f"=" * 50)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    print(f"\n🏷️ Label Distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        label_name = "Fake" if label == 1 else "Real"
        percentage = count / len(df) * 100
        print(f"  {label} ({label_name}): {count} ({percentage:.1f}%)")
    
    print(f"\n📏 Text Length Statistics:")
    df['text_length'] = df['review_text'].str.len()
    print(f"  Average length: {df['text_length'].mean():.1f} characters")
    print(f"  Min length: {df['text_length'].min()}")
    print(f"  Max length: {df['text_length'].max()}")
    print(f"  Median length: {df['text_length'].median():.1f} characters")
    
    print(f"\n⭐ Rating Distribution:")
    if 'score' in df.columns:
        rating_stats = df['score'].describe()
        print(f"  Average rating: {rating_stats['mean']:.1f}")
        print(f"  Rating range: {rating_stats['min']:.0f} - {rating_stats['max']:.0f}")
    
    print(f"\n🔤 Sample Reviews:")
    print("Real reviews:")
    real_samples = df[df['label'] == 0]['review_text'].head(2)
    for i, review in enumerate(real_samples, 1):
        print(f"  {i}. {review[:100]}...")
    
    print("\nFake reviews:")
    fake_samples = df[df['label'] == 1]['review_text'].head(2)
    for i, review in enumerate(fake_samples, 1):
        print(f"  {i}. {review[:100]}...")

if __name__ == "__main__":
    # Analyze the dataset
    analyze_chinese_fake_dataset()
    
    # Load training data
    splits = load_chinese_fake_training_data()
    if splits:
        print(f"\n✅ Successfully prepared Chinese fake training data!")
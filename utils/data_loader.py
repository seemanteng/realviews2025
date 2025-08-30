"""
Data loading utilities for training RealViews models
Supports various data formats and sources
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from utils.user_data_analyzer import UserDataAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewDataLoader:
    """Comprehensive data loader for review datasets with user metadata analysis"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.user_analyzer = UserDataAnalyzer()
        
        # Expected column mappings for different data sources
        self.column_mappings = {
            'google_reviews': {
                'text': ['review_text', 'content', 'text', 'review', 'comment'],
                'rating': ['rating', 'score', 'stars', 'star_rating'],
                'label': ['label', 'violation_type', 'category', 'class'],
                'is_violation': ['is_violation', 'violation', 'is_policy_violation']
            },
            'yelp_reviews': {
                'text': ['text', 'review_text', 'content'],
                'rating': ['stars', 'rating', 'score'],
                'useful': ['useful', 'helpful_count'],
                'funny': ['funny', 'funny_count'],
                'cool': ['cool', 'cool_count']
            },
            'amazon_reviews': {
                'text': ['reviewText', 'review_text', 'text', 'content'],
                'rating': ['overall', 'rating', 'score', 'stars'],
                'summary': ['summary', 'title', 'headline'],
                'verified': ['verified', 'verified_purchase']
            }
        }
        
        # Violation type mappings
        self.violation_mappings = {
            'advertisement': ['ad', 'advertisement', 'promotional', 'spam', 'promotion'],
            'fake': ['fake', 'synthetic', 'bot', 'generated', 'artificial'],
            'irrelevant': ['irrelevant', 'off-topic', 'unrelated', 'random'],
            'none': ['clean', 'valid', 'legitimate', 'real', 'genuine', 'none']
        }
    
    def load_csv_files(self, pattern: str = "*.csv") -> List[pd.DataFrame]:
        """Load all CSV files matching pattern from data directory"""
        csv_files = list(self.data_dir.glob(pattern))
        datasets = []
        
        logger.info(f"Found {len(csv_files)} CSV files matching '{pattern}'")
        
        for file_path in csv_files:
            try:
                logger.info(f"Loading {file_path.name}...")
                df = pd.read_csv(file_path)
                df['source_file'] = file_path.name
                datasets.append(df)
                logger.info(f"  → Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        return datasets
    
    def load_json_files(self, pattern: str = "*.json") -> List[pd.DataFrame]:
        """Load JSON files and convert to DataFrames"""
        json_files = list(self.data_dir.glob(pattern))
        datasets = []
        
        logger.info(f"Found {len(json_files)} JSON files matching '{pattern}'")
        
        for file_path in json_files:
            try:
                logger.info(f"Loading {file_path.name}...")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    if 'reviews' in data:
                        df = pd.DataFrame(data['reviews'])
                    elif 'data' in data:
                        df = pd.DataFrame(data['data'])
                    else:
                        # Assume the dict itself is the data
                        df = pd.DataFrame([data])
                
                df['source_file'] = file_path.name
                datasets.append(df)
                logger.info(f"  → Loaded {len(df)} rows from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
        
        return datasets
    
    def standardize_columns(self, df: pd.DataFrame, data_source: str = 'google_reviews') -> pd.DataFrame:
        """Standardize column names based on data source"""
        df_copy = df.copy()
        mappings = self.column_mappings.get(data_source, self.column_mappings['google_reviews'])
        
        # Standardize main columns
        for standard_col, possible_names in mappings.items():
            for possible_name in possible_names:
                if possible_name in df_copy.columns:
                    if standard_col not in df_copy.columns:
                        df_copy[standard_col] = df_copy[possible_name]
                    break
        
        return df_copy
    
    def normalize_violation_labels(self, df: pd.DataFrame, label_column: str = 'label') -> pd.DataFrame:
        """Normalize violation type labels to standard categories"""
        if label_column not in df.columns:
            logger.warning(f"Label column '{label_column}' not found")
            return df
        
        df_copy = df.copy()
        df_copy[label_column] = df_copy[label_column].astype(str).str.lower().str.strip()
        
        # Create normalized labels
        def normalize_label(label):
            label = str(label).lower().strip()
            
            for standard_type, variations in self.violation_mappings.items():
                if any(variation in label for variation in variations):
                    return standard_type
            
            # Default to analyzing the label text
            if any(word in label for word in ['promo', 'discount', 'website', 'link']):
                return 'advertisement'
            elif any(word in label for word in ['perfect', 'amazing', 'incredible', 'best ever']):
                return 'fake'
            elif any(word in label for word in ['wrong', 'unrelated', 'random']):
                return 'irrelevant'
            else:
                return 'none'
        
        df_copy['violation_type'] = df_copy[label_column].apply(normalize_label)
        df_copy['is_violation'] = df_copy['violation_type'] != 'none'
        
        return df_copy
    
    def create_training_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels for each violation type"""
        df_copy = df.copy()
        
        if 'violation_type' not in df_copy.columns:
            logger.warning("No violation_type column found. Creating from available data...")
            if 'label' in df_copy.columns:
                df_copy = self.normalize_violation_labels(df_copy)
            else:
                # Create labels based on text analysis
                df_copy['violation_type'] = 'none'
                df_copy['is_violation'] = False
        
        # Create binary columns for each violation type
        for violation_type in ['advertisement', 'fake', 'irrelevant']:
            df_copy[f'is_{violation_type}'] = (df_copy['violation_type'] == violation_type).astype(int)
        
        return df_copy
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset quality and provide statistics"""
        validation_report = {
            'total_rows': len(df),
            'missing_data': {},
            'label_distribution': {},
            'text_quality': {},
            'recommendations': []
        }
        
        # Check for missing data
        required_columns = ['text']
        for col in required_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                validation_report['missing_data'][col] = {
                    'count': missing_count,
                    'percentage': missing_pct
                }
                
                if missing_pct > 5:
                    validation_report['recommendations'].append(
                        f"High missing data in {col}: {missing_pct:.1f}%"
                    )
        
        # Label distribution
        if 'violation_type' in df.columns:
            label_dist = df['violation_type'].value_counts()
            validation_report['label_distribution'] = label_dist.to_dict()
            
            # Check for class imbalance
            min_class_pct = (label_dist.min() / len(df)) * 100
            if min_class_pct < 10:
                validation_report['recommendations'].append(
                    f"Class imbalance detected. Smallest class: {min_class_pct:.1f}%"
                )
        
        # Text quality analysis
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            validation_report['text_quality'] = {
                'avg_length': text_lengths.mean(),
                'min_length': text_lengths.min(),
                'max_length': text_lengths.max(),
                'very_short_count': (text_lengths < 10).sum(),
                'very_long_count': (text_lengths > 1000).sum()
            }
            
            short_pct = (validation_report['text_quality']['very_short_count'] / len(df)) * 100
            if short_pct > 10:
                validation_report['recommendations'].append(
                    f"Many very short texts: {short_pct:.1f}%"
                )
        
        return validation_report
    
    def prepare_training_data(self, 
                            datasets: List[pd.DataFrame],
                            test_size: float = 0.2,
                            validation_size: float = 0.1,
                            balance_classes: bool = True) -> Dict[str, pd.DataFrame]:
        """Prepare datasets for training with proper splits"""
        
        # Combine all datasets
        logger.info("Combining datasets...")
        combined_df = pd.concat(datasets, ignore_index=True)
        logger.info(f"Combined dataset size: {len(combined_df)} rows")
        
        # Ensure required columns exist
        if 'text' not in combined_df.columns:
            raise ValueError("'text' column is required but not found")
        
        # Clean and standardize
        combined_df = self.create_training_labels(combined_df)
        
        # Remove rows with missing text
        combined_df = combined_df.dropna(subset=['text'])
        
        # Balance classes if requested
        if balance_classes and 'violation_type' in combined_df.columns:
            combined_df = self._balance_classes(combined_df)
        
        # Create splits
        logger.info("Creating train/validation/test splits...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            combined_df,
            test_size=test_size,
            stratify=combined_df.get('violation_type'),
            random_state=42
        )
        
        # Second split: train vs validation
        val_size_adjusted = validation_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df.get('violation_type'),
            random_state=42
        )
        
        splits = {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'full': combined_df
        }
        
        # Log split statistics
        for split_name, split_df in splits.items():
            if split_name != 'full':
                logger.info(f"{split_name.capitalize()} set: {len(split_df)} rows")
                if 'violation_type' in split_df.columns:
                    dist = split_df['violation_type'].value_counts()
                    logger.info(f"  Label distribution: {dist.to_dict()}")
        
        return splits
    
    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance classes using undersampling/oversampling"""
        if 'violation_type' not in df.columns:
            return df
        
        logger.info("Balancing classes...")
        
        # Get class counts
        class_counts = df['violation_type'].value_counts()
        target_count = int(class_counts.median())  # Use median as target
        
        balanced_dfs = []
        for violation_type in class_counts.index:
            class_df = df[df['violation_type'] == violation_type]
            current_count = len(class_df)
            
            if current_count > target_count:
                # Undersample
                sampled_df = class_df.sample(n=target_count, random_state=42)
                logger.info(f"  {violation_type}: {current_count} → {target_count} (undersampled)")
            elif current_count < target_count:
                # Oversample (with replacement)
                sampled_df = class_df.sample(n=target_count, replace=True, random_state=42)
                logger.info(f"  {violation_type}: {current_count} → {target_count} (oversampled)")
            else:
                sampled_df = class_df
                logger.info(f"  {violation_type}: {current_count} (no change)")
            
            balanced_dfs.append(sampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
        logger.info(f"Balanced dataset size: {len(balanced_df)} rows")
        
        return balanced_df
    
    def save_processed_data(self, splits: Dict[str, pd.DataFrame], output_dir: str = "data/processed"):
        """Save processed training data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to {output_path}")
        
        for split_name, split_df in splits.items():
            if split_name != 'full':  # Don't save the full dataset separately
                file_path = output_path / f"{split_name}_data.csv"
                split_df.to_csv(file_path, index=False)
                logger.info(f"  Saved {split_name}: {len(split_df)} rows → {file_path}")
        
        # Save metadata
        metadata = {
            'total_samples': len(splits['full']),
            'train_samples': len(splits['train']),
            'validation_samples': len(splits['validation']),
            'test_samples': len(splits['test']),
            'features': list(splits['full'].columns),
            'label_distribution': splits['full']['violation_type'].value_counts().to_dict() if 'violation_type' in splits['full'].columns else {}
        }
        
        metadata_path = output_path / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  Saved metadata → {metadata_path}")
    
    def load_example_datasets(self):
        """Download and prepare example datasets for demonstration"""
        logger.info("Creating example datasets...")
        
        # Create sample data files to demonstrate the loading process
        examples = {
            'google_reviews_sample.csv': self._create_google_sample(),
            'yelp_reviews_sample.json': self._create_yelp_sample(),
            'amazon_reviews_sample.csv': self._create_amazon_sample()
        }
        
        for filename, data in examples.items():
            file_path = self.data_dir / filename
            
            if filename.endswith('.csv'):
                data.to_csv(file_path, index=False)
            elif filename.endswith('.json'):
                data.to_json(file_path, orient='records', indent=2)
            
            logger.info(f"Created example file: {file_path}")
    
    def _create_google_sample(self) -> pd.DataFrame:
        """Create sample Google Reviews dataset"""
        return pd.DataFrame([
            {"review_text": "Great food and excellent service. The pasta was delicious!", "rating": 5, "label": "none"},
            {"review_text": "Amazing restaurant! Visit our website www.example.com for 50% off!", "rating": 5, "label": "advertisement"},
            {"review_text": "Perfect perfect perfect! Best restaurant ever! Amazing incredible!", "rating": 5, "label": "fake"},
            {"review_text": "This review is about a different restaurant in another city.", "rating": 3, "label": "irrelevant"},
            {"review_text": "Good location, decent food. Staff could be friendlier.", "rating": 3, "label": "none"},
            {"review_text": "Terrible experience. Food was cold and service was slow.", "rating": 1, "label": "none"}
        ])
    
    def _create_yelp_sample(self) -> pd.DataFrame:
        """Create sample Yelp dataset"""
        return pd.DataFrame([
            {"text": "Love this place! Great atmosphere and friendly staff.", "stars": 4, "useful": 2, "funny": 0, "cool": 1},
            {"text": "Check out our special deals at restaurantdeals.com!", "stars": 5, "useful": 0, "funny": 0, "cool": 0},
            {"text": "Absolutely phenomenal extraordinary perfect experience!", "stars": 5, "useful": 0, "funny": 0, "cool": 0},
            {"text": "Wrong restaurant - I was looking for the hotel next door.", "stars": 1, "useful": 0, "funny": 0, "cool": 0}
        ])
    
    def _create_amazon_sample(self) -> pd.DataFrame:
        """Create sample Amazon Reviews dataset"""
        return pd.DataFrame([
            {"reviewText": "High quality product, fast shipping. Recommended!", "overall": 5, "summary": "Great purchase", "verified": True},
            {"reviewText": "Buy more products at our store! Special discount available!", "overall": 5, "summary": "Amazing", "verified": False},
            {"reviewText": "Perfect perfect perfect! Best product ever made!", "overall": 5, "summary": "Perfect", "verified": False}
        ])
    
    def load_user_metadata(self, accounts_file: str = "accounts.csv", reviews_file: str = "account_reviews.csv") -> Dict[str, Any]:
        """
        Load and analyze user metadata for fake review detection
        """
        accounts_path = self.data_dir / accounts_file
        reviews_path = self.data_dir / reviews_file
        
        if not accounts_path.exists():
            logger.warning(f"Accounts file not found: {accounts_path}")
            return {}
        
        if not reviews_path.exists():
            logger.warning(f"Reviews file not found: {reviews_path}")
            return {}
        
        logger.info("🔍 Loading user metadata for fake review analysis...")
        
        try:
            # Load the data
            accounts_df = pd.read_csv(accounts_path)
            reviews_df = pd.read_csv(reviews_path)
            
            logger.info(f"  Loaded {len(accounts_df)} accounts and {len(reviews_df)} reviews")
            
            # Analyze user patterns
            user_analysis = self.user_analyzer.generate_user_suspicion_report(accounts_df, reviews_df)
            
            return user_analysis
            
        except Exception as e:
            logger.error(f"Failed to load user metadata: {e}")
            return {}
    
    def integrate_user_features(self, review_df: pd.DataFrame, user_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        Integrate user suspicion features into review dataset for training
        """
        if not user_analysis or 'final_suspicion_scores' not in user_analysis:
            logger.warning("No user analysis available for feature integration")
            return review_df
        
        logger.info("🔗 Integrating user suspicion features...")
        
        enhanced_df = review_df.copy()
        suspicion_scores = user_analysis['final_suspicion_scores']
        
        # Map account_id to account suspicion features
        if 'account_id' in enhanced_df.columns:
            enhanced_df = enhanced_df.merge(
                suspicion_scores[['account_id', 'composite_suspicion_score', 'suspicion_level', 
                                'has_burst_activity', 'account_suspicion_score']],
                on='account_id',
                how='left'
            )
            
            # Fill missing values for reviews without account data
            feature_columns = ['composite_suspicion_score', 'account_suspicion_score']
            enhanced_df[feature_columns] = enhanced_df[feature_columns].fillna(0.0)
            enhanced_df['has_burst_activity'] = enhanced_df['has_burst_activity'].fillna(False)
            enhanced_df['suspicion_level'] = enhanced_df['suspicion_level'].astype(str).fillna('unknown')
            
            # Create additional features based on user analysis
            enhanced_df['is_high_suspicion_account'] = (
                enhanced_df['composite_suspicion_score'] > 0.6
            )
            enhanced_df['is_burst_account'] = enhanced_df['has_burst_activity']
            
            logger.info(f"  Added user suspicion features to {len(enhanced_df)} reviews")
            logger.info(f"  High suspicion accounts: {enhanced_df['is_high_suspicion_account'].sum()}")
            logger.info(f"  Burst activity accounts: {enhanced_df['is_burst_account'].sum()}")
        
        else:
            logger.warning("No account_id column found - cannot integrate user features")
        
        return enhanced_df
    
    def create_user_aware_training_data(self, 
                                      datasets: List[pd.DataFrame],
                                      accounts_file: str = "accounts.csv",
                                      reviews_file: str = "account_reviews.csv",
                                      **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Create training data enhanced with user metadata features
        """
        logger.info("🚀 Creating user-aware training dataset...")
        
        # Load and analyze user metadata
        user_analysis = self.load_user_metadata(accounts_file, reviews_file)
        
        # Prepare basic training data
        splits = self.prepare_training_data(datasets, **kwargs)
        
        # Integrate user features if analysis is available
        if user_analysis:
            logger.info("✨ Enhancing training data with user suspicion features...")
            
            for split_name, split_df in splits.items():
                if split_name != 'full':
                    splits[split_name] = self.integrate_user_features(split_df, user_analysis)
            
            # Store user analysis for later reference
            splits['user_analysis'] = user_analysis
        
        return splits
    
    def generate_user_metadata_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive user metadata analysis report
        """
        logger.info("📊 Generating user metadata analysis report...")
        
        user_analysis = self.load_user_metadata()
        
        if not user_analysis:
            return {"error": "No user metadata available"}
        
        # Create a comprehensive report
        report = {
            "executive_summary": user_analysis.get('summary', {}),
            "recommendations": user_analysis.get('recommendations', []),
            "detailed_analysis": {
                "burst_patterns": user_analysis.get('burst_analysis', {}),
                "device_patterns": user_analysis.get('device_analysis', {}),
                "referrer_analysis": user_analysis.get('referrer_analysis', {})
            },
            "suspicious_accounts": [],
            "high_risk_indicators": []
        }
        
        # Extract high-risk accounts
        if 'final_suspicion_scores' in user_analysis:
            high_suspicion = user_analysis['final_suspicion_scores'][
                user_analysis['final_suspicion_scores']['composite_suspicion_score'] > 0.7
            ]
            report['suspicious_accounts'] = high_suspicion['account_id'].tolist()
        
        # Identify key risk indicators
        summary = user_analysis.get('summary', {})
        if summary.get('fake_accounts_percentage', 0) > 20:
            report['high_risk_indicators'].append("High percentage of fake accounts detected")
        if summary.get('burst_accounts_percentage', 0) > 15:
            report['high_risk_indicators'].append("Significant review burst activity")
        if summary.get('reviews_with_links', 0) > summary.get('total_reviews', 1) * 0.1:
            report['high_risk_indicators'].append("High volume of reviews with links")
        
        return report

def main():
    """Example usage of the data loader"""
    loader = ReviewDataLoader()
    
    # Create example datasets if none exist
    if not list(loader.data_dir.glob("*.csv")) and not list(loader.data_dir.glob("*.json")):
        print("No data files found. Creating example datasets...")
        loader.load_example_datasets()
    
    # Load all available data
    csv_datasets = loader.load_csv_files()
    json_datasets = loader.load_json_files()
    
    all_datasets = csv_datasets + json_datasets
    
    if not all_datasets:
        print("No datasets loaded. Please add data files to data/raw/")
        return
    
    # Standardize and prepare data
    standardized_datasets = []
    for i, df in enumerate(all_datasets):
        source_type = 'google_reviews'  # Default, could be auto-detected
        df_std = loader.standardize_columns(df, source_type)
        df_std = loader.normalize_violation_labels(df_std)
        standardized_datasets.append(df_std)
        
        # Show validation report
        validation = loader.validate_dataset(df_std)
        print(f"\nDataset {i+1} validation:")
        print(f"  Total rows: {validation['total_rows']}")
        print(f"  Label distribution: {validation['label_distribution']}")
        for rec in validation['recommendations'][:3]:  # Show top 3 recommendations
            print(f"  ⚠️  {rec}")
    
    # Prepare training splits
    splits = loader.prepare_training_data(standardized_datasets)
    
    # Save processed data
    loader.save_processed_data(splits)
    
    print(f"\n✅ Data preparation complete!")
    print(f"   Train: {len(splits['train'])} samples")
    print(f"   Validation: {len(splits['validation'])} samples") 
    print(f"   Test: {len(splits['test'])} samples")
    print(f"   Data saved to: data/processed/")

if __name__ == "__main__":
    main()
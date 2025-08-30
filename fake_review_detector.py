#!/usr/bin/env python3
"""
Fake Review Detection System using Metadata Analysis

This system uses user metadata to detect fake reviews through:
1. Submission time gap analysis
2. Device/browser fingerprinting
3. Burst analysis for review storms
4. User behavior patterns
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class FakeReviewDetector:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.reviews_df = None
        self.features_df = None
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, data_path=None):
        """Load review data from JSON file"""
        if data_path:
            self.data_path = data_path
            
        print(f"Loading data from {self.data_path}...")
        reviews = []
        
        with open(self.data_path, 'r') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    reviews.append(review)
                except json.JSONDecodeError:
                    continue
                    
        self.reviews_df = pd.DataFrame(reviews)
        print(f"Loaded {len(self.reviews_df)} reviews")
        return self.reviews_df
    
    def extract_temporal_features(self):
        """Extract time-based features for detecting suspicious patterns"""
        print("Extracting temporal features...")
        
        # Convert timestamp to datetime
        self.reviews_df['datetime'] = pd.to_datetime(self.reviews_df['time'], unit='ms')
        self.reviews_df['hour'] = self.reviews_df['datetime'].dt.hour
        self.reviews_df['day_of_week'] = self.reviews_df['datetime'].dt.dayofweek
        self.reviews_df['day_of_month'] = self.reviews_df['datetime'].dt.day
        
        temporal_features = []
        
        for idx, review in self.reviews_df.iterrows():
            features = {
                'review_id': idx,
                'user_id': review['user_id'],
                'gmap_id': review['gmap_id'],
                'hour_of_day': review['hour'],
                'day_of_week': review['day_of_week'],
                'day_of_month': review['day_of_month']
            }
            
            # Calculate time gaps for this user
            user_reviews = self.reviews_df[self.reviews_df['user_id'] == review['user_id']].sort_values('time')
            if len(user_reviews) > 1:
                time_diffs = user_reviews['time'].diff().fillna(0)
                features['avg_time_gap_hours'] = time_diffs.mean() / (1000 * 60 * 60)
                features['min_time_gap_hours'] = time_diffs[time_diffs > 0].min() / (1000 * 60 * 60) if len(time_diffs[time_diffs > 0]) > 0 else 0
                features['user_review_count'] = len(user_reviews)
            else:
                features['avg_time_gap_hours'] = 0
                features['min_time_gap_hours'] = 0
                features['user_review_count'] = 1
                
            # Calculate business-specific patterns
            business_reviews = self.reviews_df[self.reviews_df['gmap_id'] == review['gmap_id']].sort_values('time')
            features['business_review_count'] = len(business_reviews)
            
            temporal_features.append(features)
            
        return pd.DataFrame(temporal_features)
    
    def extract_user_fingerprint_features(self):
        """Extract device/browser fingerprinting-like features from user behavior"""
        print("Extracting user fingerprinting features...")
        
        user_patterns = {}
        
        for user_id, user_reviews in self.reviews_df.groupby('user_id'):
            patterns = {
                'total_reviews': len(user_reviews),
                'unique_businesses': user_reviews['gmap_id'].nunique(),
                'avg_rating': user_reviews['rating'].mean(),
                'rating_variance': user_reviews['rating'].var(),
                'has_text_ratio': (user_reviews['text'].notna()).mean(),
                'has_pics_ratio': (user_reviews['pics'].notna()).mean(),
                'response_received_ratio': (user_reviews['resp'].notna()).mean(),
                'name_length': len(str(user_reviews.iloc[0]['name'])),
                'name_has_numbers': any(char.isdigit() for char in str(user_reviews.iloc[0]['name'])),
                'activity_span_days': (user_reviews['time'].max() - user_reviews['time'].min()) / (1000 * 60 * 60 * 24)
            }
            
            # Rating distribution pattern
            rating_counts = user_reviews['rating'].value_counts()
            for rating in range(1, 6):
                patterns[f'rating_{rating}_ratio'] = rating_counts.get(rating, 0) / len(user_reviews)
            
            user_patterns[user_id] = patterns
            
        return pd.DataFrame.from_dict(user_patterns, orient='index').reset_index().rename(columns={'index': 'user_id'})
    
    def detect_review_bursts(self, time_window_hours=24, min_reviews=3):
        """Detect suspicious review storms/bursts"""
        print(f"Detecting review bursts (>{min_reviews} reviews in {time_window_hours}h window)...")
        
        burst_features = []
        window_ms = time_window_hours * 60 * 60 * 1000
        
        for gmap_id, business_reviews in self.reviews_df.groupby('gmap_id'):
            business_reviews = business_reviews.sort_values('time').reset_index()
            
            for i, review in business_reviews.iterrows():
                # Count reviews in time window
                window_start = review['time'] - window_ms
                window_end = review['time'] + window_ms
                
                window_reviews = business_reviews[
                    (business_reviews['time'] >= window_start) & 
                    (business_reviews['time'] <= window_end)
                ]
                
                features = {
                    'review_id': review['index'],
                    'reviews_in_window': len(window_reviews),
                    'unique_users_in_window': window_reviews['user_id'].nunique(),
                    'avg_rating_in_window': window_reviews['rating'].mean(),
                    'rating_variance_in_window': window_reviews['rating'].var(),
                    'is_burst': len(window_reviews) >= min_reviews,
                    'user_diversity_ratio': window_reviews['user_id'].nunique() / len(window_reviews),
                    'text_reviews_ratio': (window_reviews['text'].notna()).mean()
                }
                
                burst_features.append(features)
                
        return pd.DataFrame(burst_features)
    
    def create_anomaly_features(self):
        """Create features that help identify anomalous behavior"""
        print("Creating anomaly detection features...")
        
        anomaly_features = []
        
        for idx, review in self.reviews_df.iterrows():
            user_id = review['user_id']
            gmap_id = review['gmap_id']
            
            # User behavior anomalies
            user_reviews = self.reviews_df[self.reviews_df['user_id'] == user_id]
            all_users = self.reviews_df['user_id'].value_counts()
            
            # Business review anomalies  
            business_reviews = self.reviews_df[self.reviews_df['gmap_id'] == gmap_id]
            
            features = {
                'review_id': idx,
                'user_activity_percentile': (all_users > len(user_reviews)).mean(),
                'user_rating_deviation': abs(review['rating'] - user_reviews['rating'].mean()),
                'business_rating_deviation': abs(review['rating'] - business_reviews['rating'].mean()),
                'is_extreme_rating': review['rating'] in [1, 5],
                'text_length': len(str(review['text'])) if review['text'] else 0,
                'is_empty_text': review['text'] is None or str(review['text']).strip() == '',
                'has_business_response': review['resp'] is not None,
                'user_name_similarity_score': self._calculate_name_similarity(user_id),
            }
            
            anomaly_features.append(features)
            
        return pd.DataFrame(anomaly_features)
    
    def _calculate_name_similarity(self, user_id):
        """Calculate similarity of user name to other names (simple heuristic)"""
        user_name = self.reviews_df[self.reviews_df['user_id'] == user_id]['name'].iloc[0]
        if not isinstance(user_name, str):
            return 0
            
        # Simple similarity check based on common patterns in fake names
        name_patterns = ['user', 'test', 'fake', 'bot', 'admin']
        similarity_score = sum(1 for pattern in name_patterns if pattern.lower() in user_name.lower())
        
        return similarity_score
    
    def combine_all_features(self):
        """Combine all extracted features into final feature set"""
        print("Combining all features...")
        
        # Extract all feature sets
        temporal_features = self.extract_temporal_features()
        user_features = self.extract_user_fingerprint_features()
        burst_features = self.detect_review_bursts()
        anomaly_features = self.create_anomaly_features()
        
        # Merge features
        features = temporal_features.merge(
            user_features, on='user_id', how='left'
        ).merge(
            burst_features, on='review_id', how='left'
        ).merge(
            anomaly_features, on='review_id', how='left'
        )
        
        # Fill missing values
        features = features.fillna(0)
        
        self.features_df = features
        print(f"Created feature matrix with {len(features)} rows and {len(features.columns)} columns")
        return features
    
    def create_synthetic_labels(self):
        """Create synthetic labels for training based on suspicious patterns"""
        print("Creating synthetic labels based on suspicious patterns...")
        
        if self.features_df is None:
            self.combine_all_features()
            
        # Define suspicious behavior patterns
        conditions = [
            # Very short time gaps (bot-like behavior)
            (self.features_df['min_time_gap_hours'] < 0.1) & (self.features_df['user_review_count'] > 2),
            
            # Review bursts with low diversity
            (self.features_df['is_burst'] == True) & (self.features_df['user_diversity_ratio'] < 0.3),
            
            # Extreme ratings with empty text
            (self.features_df['is_extreme_rating'] == True) & (self.features_df['is_empty_text'] == True),
            
            # Very high activity users with uniform ratings
            (self.features_df['total_reviews'] > 50) & (self.features_df['rating_variance'] < 0.5),
            
            # Users with suspicious name patterns
            (self.features_df['user_name_similarity_score'] > 0),
            
            # Multiple reviews on same day with similar ratings
            (self.features_df['reviews_in_window'] > 5) & (self.features_df['rating_variance_in_window'] < 1.0)
        ]
        
        # Combine conditions (if any condition is true, mark as suspicious)
        suspicious_mask = np.zeros(len(self.features_df), dtype=bool)
        for condition in conditions:
            suspicious_mask |= condition
            
        self.features_df['is_fake'] = suspicious_mask.astype(int)
        
        print(f"Labeled {suspicious_mask.sum()} reviews as potentially fake ({suspicious_mask.mean():.2%})")
        return self.features_df['is_fake']
    
    def train_model(self):
        """Train fake review detection model"""
        print("Training fake review detection model...")
        
        if self.features_df is None:
            self.combine_all_features()
            
        # Create labels
        labels = self.create_synthetic_labels()
        
        # Select feature columns (exclude identifiers and target)
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['review_id', 'user_id', 'gmap_id', 'is_fake']]
        
        X = self.features_df[feature_cols]
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        return self.model, feature_importance
    
    def detect_anomalies(self, contamination=0.1):
        """Use Isolation Forest for unsupervised anomaly detection"""
        print(f"Running Isolation Forest anomaly detection (contamination={contamination})...")
        
        if self.features_df is None:
            self.combine_all_features()
            
        # Select feature columns
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['review_id', 'user_id', 'gmap_id', 'is_fake']]
        
        X = self.features_df[feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)
        
        self.features_df['anomaly_score'] = anomaly_scores
        self.features_df['is_anomaly'] = (anomaly_labels == -1).astype(int)
        
        print(f"Detected {(anomaly_labels == -1).sum()} anomalies ({(anomaly_labels == -1).mean():.2%})")
        
        return anomaly_labels, anomaly_scores
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*50)
        print("FAKE REVIEW DETECTION REPORT")
        print("="*50)
        
        if self.features_df is None:
            print("No analysis performed yet. Run combine_all_features() first.")
            return
            
        # Basic statistics
        print(f"\nDataset Overview:")
        print(f"Total reviews: {len(self.reviews_df)}")
        print(f"Unique users: {self.reviews_df['user_id'].nunique()}")
        print(f"Unique businesses: {self.reviews_df['gmap_id'].nunique()}")
        print(f"Date range: {self.reviews_df['datetime'].min()} to {self.reviews_df['datetime'].max()}")
        
        # Temporal patterns
        print(f"\nTemporal Patterns:")
        hourly_dist = self.reviews_df['hour'].value_counts().sort_index()
        peak_hour = hourly_dist.idxmax()
        print(f"Peak review hour: {peak_hour}:00 ({hourly_dist[peak_hour]} reviews)")
        
        # User behavior
        print(f"\nUser Behavior:")
        user_review_counts = self.reviews_df['user_id'].value_counts()
        print(f"Users with >10 reviews: {(user_review_counts > 10).sum()}")
        print(f"Users with >50 reviews: {(user_review_counts > 50).sum()}")
        print(f"Most active user: {user_review_counts.iloc[0]} reviews")
        
        # Rating distribution
        print(f"\nRating Distribution:")
        rating_dist = self.reviews_df['rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            print(f"  {rating} stars: {count} ({count/len(self.reviews_df):.1%})")
            
        # Suspicious patterns detected
        if 'is_fake' in self.features_df.columns:
            fake_count = self.features_df['is_fake'].sum()
            print(f"\nSuspicious Reviews Detected: {fake_count} ({fake_count/len(self.features_df):.1%})")
            
        if 'is_anomaly' in self.features_df.columns:
            anomaly_count = self.features_df['is_anomaly'].sum()
            print(f"Anomalies Detected: {anomaly_count} ({anomaly_count/len(self.features_df):.1%})")
            
        print("\n" + "="*50)

# Usage example and main execution
if __name__ == "__main__":
    # Initialize detector
    detector = FakeReviewDetector()
    
    # Load and analyze data
    data_path = "data/raw/review-Mississippi_10.json"
    detector.load_data(data_path)
    
    # Extract features and train model
    detector.combine_all_features()
    model, feature_importance = detector.train_model()
    
    # Run anomaly detection
    detector.detect_anomalies()
    
    # Generate comprehensive report
    detector.generate_report()
    
    # Save results
    output_path = "fake_review_analysis_results.csv"
    detector.features_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
#!/usr/bin/env python3
"""
Efficient Fake Review Detection Training Script
Processes large datasets in chunks for better memory management
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from collections import defaultdict

def load_reviews_sample(file_path, max_reviews=50000):
    """Load a sample of reviews for faster processing"""
    print(f"Loading up to {max_reviews} reviews from {file_path}...")
    
    reviews = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_reviews:
                break
            try:
                review = json.loads(line.strip())
                reviews.append(review)
            except json.JSONDecodeError:
                continue
                
    df = pd.DataFrame(reviews)
    print(f"Loaded {len(df)} reviews")
    return df

def extract_features(df):
    """Extract comprehensive features for fake review detection"""
    print("Extracting features...")
    
    # Convert timestamp
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    features_list = []
    
    # Group operations for efficiency
    user_stats = df.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std'],
        'time': ['min', 'max'],
        'text': lambda x: (x.notna()).mean(),
        'gmap_id': 'nunique'
    }).round(3)
    
    business_stats = df.groupby('gmap_id').agg({
        'rating': ['count', 'mean', 'std'],
        'time': ['min', 'max']
    }).round(3)
    
    for idx, review in df.iterrows():
        user_id = review['user_id']
        gmap_id = review['gmap_id']
        
        # Get pre-computed stats
        user_review_count = user_stats.loc[user_id, ('rating', 'count')]
        user_avg_rating = user_stats.loc[user_id, ('rating', 'mean')]
        user_rating_std = user_stats.loc[user_id, ('rating', 'std')] or 0
        user_text_ratio = user_stats.loc[user_id, ('text', '<lambda>')]
        user_business_count = user_stats.loc[user_id, ('gmap_id', 'nunique')]
        
        business_review_count = business_stats.loc[gmap_id, ('rating', 'count')]
        business_avg_rating = business_stats.loc[gmap_id, ('rating', 'mean')]
        
        # Calculate time-based features
        user_reviews = df[df['user_id'] == user_id].sort_values('time')
        if len(user_reviews) > 1:
            time_diffs = user_reviews['time'].diff().dropna()
            min_gap_hours = time_diffs.min() / (1000 * 60 * 60) if len(time_diffs) > 0 else 0
            avg_gap_hours = time_diffs.mean() / (1000 * 60 * 60) if len(time_diffs) > 0 else 0
        else:
            min_gap_hours = 0
            avg_gap_hours = 0
            
        # Burst detection (simplified)
        business_reviews = df[df['gmap_id'] == gmap_id]
        time_window = 24 * 60 * 60 * 1000  # 24 hours in milliseconds
        nearby_reviews = business_reviews[
            (business_reviews['time'] >= review['time'] - time_window) &
            (business_reviews['time'] <= review['time'] + time_window)
        ]
        
        features = {
            # User behavior features
            'user_review_count': user_review_count,
            'user_avg_rating': user_avg_rating,
            'user_rating_std': user_rating_std,
            'user_business_diversity': user_business_count,
            'user_text_ratio': user_text_ratio,
            
            # Temporal features
            'hour_of_day': review['hour'],
            'day_of_week': review['day_of_week'],
            'min_gap_hours': min_gap_hours,
            'avg_gap_hours': avg_gap_hours,
            
            # Review content features
            'rating': review['rating'],
            'is_extreme_rating': 1 if review['rating'] in [1, 5] else 0,
            'has_text': 1 if pd.notna(review['text']) and str(review['text']).strip() != '' else 0,
            'text_length': len(str(review['text'])) if pd.notna(review['text']) else 0,
            'has_pics': 1 if review['pics'] is not None else 0,
            'has_response': 1 if review['resp'] is not None else 0,
            
            # Business context features
            'business_review_count': business_review_count,
            'business_avg_rating': business_avg_rating,
            'rating_deviation_from_business': abs(review['rating'] - business_avg_rating),
            'rating_deviation_from_user': abs(review['rating'] - user_avg_rating),
            
            # Burst/storm features
            'reviews_in_24h_window': len(nearby_reviews),
            'unique_users_in_window': nearby_reviews['user_id'].nunique(),
            'user_diversity_ratio': nearby_reviews['user_id'].nunique() / len(nearby_reviews) if len(nearby_reviews) > 0 else 1,
            
            # Suspicious patterns
            'very_active_user': 1 if user_review_count > 50 else 0,
            'single_business_user': 1 if user_business_count == 1 else 0,
            'rapid_reviewer': 1 if min_gap_hours < 1 and user_review_count > 3 else 0,
            'uniform_rater': 1 if user_rating_std < 0.5 and user_review_count > 5 else 0,
        }
        
        features_list.append(features)
        
        if idx % 5000 == 0:
            print(f"Processed {idx} reviews...")
    
    return pd.DataFrame(features_list)

def create_labels(features_df):
    """Create synthetic labels based on suspicious behavior patterns"""
    print("Creating labels based on suspicious patterns...")
    
    # Define multiple suspicious behavior indicators
    suspicious_conditions = [
        # Rapid reviewers with minimal gaps
        (features_df['rapid_reviewer'] == 1),
        
        # Users with uniform ratings and high activity
        (features_df['uniform_rater'] == 1),
        
        # Extreme ratings without text
        (features_df['is_extreme_rating'] == 1) & (features_df['has_text'] == 0),
        
        # Burst activity with low user diversity
        (features_df['reviews_in_24h_window'] >= 5) & (features_df['user_diversity_ratio'] < 0.4),
        
        # Very active users with single business focus
        (features_df['very_active_user'] == 1) & (features_df['single_business_user'] == 1),
        
        # Outlier patterns
        (features_df['user_review_count'] > 100) & (features_df['user_rating_std'] < 0.3),
    ]
    
    # Combine conditions
    is_suspicious = np.zeros(len(features_df), dtype=bool)
    for condition in suspicious_conditions:
        is_suspicious |= condition
    
    labels = is_suspicious.astype(int)
    print(f"Labeled {labels.sum()} reviews as suspicious ({labels.mean():.2%})")
    
    return labels

def train_model(features_df, labels):
    """Train the fake review detection model"""
    print("Training model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features_df.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Detailed classification report
    y_pred = model.predict(X_test_scaled)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, scaler, feature_importance

def save_model(model, scaler, feature_importance):
    """Save trained model and components"""
    print("Saving model...")
    
    # Save model components
    with open('fake_review_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_importance': feature_importance
        }, f)
    
    print("Model saved to fake_review_model.pkl")

def main():
    """Main training pipeline"""
    print("Starting Fake Review Detection Training Pipeline")
    print("="*50)
    
    # Load data
    data_path = "data/raw/review-Mississippi_10.json"
    reviews_df = load_reviews_sample(data_path, max_reviews=20000)  # Reduced for faster processing
    
    # Extract features
    features_df = extract_features(reviews_df)
    
    # Create labels
    labels = create_labels(features_df)
    
    # Train model
    model, scaler, feature_importance = train_model(features_df, labels)
    
    # Save model
    save_model(model, scaler, feature_importance)
    
    print("\nTraining completed successfully!")
    print(f"Model can now detect fake reviews using {len(features_df.columns)} features")

if __name__ == "__main__":
    main()
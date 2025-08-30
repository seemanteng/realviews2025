#!/usr/bin/env python3
"""
Fake Review Detection Utility
Uses the trained model to detect fake reviews in new data
"""

import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

class FakeReviewPredictor:
    def __init__(self, model_path='fake_review_model.pkl'):
        """Load the trained model"""
        print(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        
        print("Model loaded successfully!")
        print(f"Model uses {len(self.feature_importance)} features")
        
    def extract_features_single(self, reviews_df):
        """Extract features from review data for prediction"""
        print(f"Extracting features for {len(reviews_df)} reviews...")
        
        # Convert timestamp
        reviews_df['datetime'] = pd.to_datetime(reviews_df['time'], unit='ms')
        reviews_df['hour'] = reviews_df['datetime'].dt.hour
        reviews_df['day_of_week'] = reviews_df['datetime'].dt.dayofweek
        
        features_list = []
        
        # Pre-compute stats for efficiency
        user_stats = reviews_df.groupby('user_id').agg({
            'rating': ['count', 'mean', 'std'],
            'time': ['min', 'max'],
            'text': lambda x: (x.notna()).mean(),
            'gmap_id': 'nunique'
        }).round(3)
        
        business_stats = reviews_df.groupby('gmap_id').agg({
            'rating': ['count', 'mean', 'std'],
            'time': ['min', 'max']
        }).round(3)
        
        for idx, review in reviews_df.iterrows():
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
            user_reviews = reviews_df[reviews_df['user_id'] == user_id].sort_values('time')
            if len(user_reviews) > 1:
                time_diffs = user_reviews['time'].diff().dropna()
                min_gap_hours = time_diffs.min() / (1000 * 60 * 60) if len(time_diffs) > 0 else 0
                avg_gap_hours = time_diffs.mean() / (1000 * 60 * 60) if len(time_diffs) > 0 else 0
            else:
                min_gap_hours = 0
                avg_gap_hours = 0
                
            # Burst detection
            business_reviews = reviews_df[reviews_df['gmap_id'] == gmap_id]
            time_window = 24 * 60 * 60 * 1000  # 24 hours
            nearby_reviews = business_reviews[
                (business_reviews['time'] >= review['time'] - time_window) &
                (business_reviews['time'] <= review['time'] + time_window)
            ]
            
            features = {
                'user_review_count': user_review_count,
                'user_avg_rating': user_avg_rating,
                'user_rating_std': user_rating_std,
                'user_business_diversity': user_business_count,
                'user_text_ratio': user_text_ratio,
                'hour_of_day': review['hour'],
                'day_of_week': review['day_of_week'],
                'min_gap_hours': min_gap_hours,
                'avg_gap_hours': avg_gap_hours,
                'rating': review['rating'],
                'is_extreme_rating': 1 if review['rating'] in [1, 5] else 0,
                'has_text': 1 if pd.notna(review['text']) and str(review['text']).strip() != '' else 0,
                'text_length': len(str(review['text'])) if pd.notna(review['text']) else 0,
                'has_pics': 1 if review['pics'] is not None else 0,
                'has_response': 1 if review['resp'] is not None else 0,
                'business_review_count': business_review_count,
                'business_avg_rating': business_avg_rating,
                'rating_deviation_from_business': abs(review['rating'] - business_avg_rating),
                'rating_deviation_from_user': abs(review['rating'] - user_avg_rating),
                'reviews_in_24h_window': len(nearby_reviews),
                'unique_users_in_window': nearby_reviews['user_id'].nunique(),
                'user_diversity_ratio': nearby_reviews['user_id'].nunique() / len(nearby_reviews) if len(nearby_reviews) > 0 else 1,
                'very_active_user': 1 if user_review_count > 50 else 0,
                'single_business_user': 1 if user_business_count == 1 else 0,
                'rapid_reviewer': 1 if min_gap_hours < 1 and user_review_count > 3 else 0,
                'uniform_rater': 1 if user_rating_std < 0.5 and user_review_count > 5 else 0,
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def predict(self, reviews_df):
        """Predict fake reviews in the dataset"""
        # Extract features
        features_df = self.extract_features_single(reviews_df)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]  # Probability of being fake
        
        # Add results to original dataframe
        results_df = reviews_df.copy()
        results_df['is_fake_prediction'] = predictions
        results_df['fake_probability'] = probabilities
        results_df['risk_level'] = pd.cut(probabilities, 
                                         bins=[0, 0.3, 0.7, 1.0], 
                                         labels=['Low', 'Medium', 'High'])
        
        return results_df, features_df
    
    def analyze_suspicious_patterns(self, results_df):
        """Analyze patterns in suspicious reviews"""
        print("\n" + "="*50)
        print("SUSPICIOUS REVIEW ANALYSIS")
        print("="*50)
        
        fake_reviews = results_df[results_df['is_fake_prediction'] == 1]
        
        print(f"Total reviews analyzed: {len(results_df)}")
        print(f"Suspicious reviews detected: {len(fake_reviews)} ({len(fake_reviews)/len(results_df):.1%})")
        
        if len(fake_reviews) > 0:
            print(f"\nRisk Level Distribution:")
            risk_counts = results_df['risk_level'].value_counts()
            for risk, count in risk_counts.items():
                print(f"  {risk} Risk: {count} ({count/len(results_df):.1%})")
            
            print(f"\nSuspicious Review Characteristics:")
            print(f"  Average rating: {fake_reviews['rating'].mean():.1f}")
            print(f"  Empty text ratio: {(fake_reviews['text'].isna() | (fake_reviews['text'] == '')).mean():.1%}")
            print(f"  Extreme ratings (1 or 5): {(fake_reviews['rating'].isin([1, 5])).mean():.1%}")
            
            print(f"\nTop Suspicious Users:")
            suspicious_users = fake_reviews['user_id'].value_counts().head()
            for user, count in suspicious_users.items():
                user_name = fake_reviews[fake_reviews['user_id'] == user]['name'].iloc[0]
                print(f"  {user_name}: {count} suspicious reviews")
        
        return fake_reviews

def load_sample_data(file_path, max_reviews=5000):
    """Load sample data for testing"""
    print(f"Loading sample data from {file_path}...")
    
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
                
    return pd.DataFrame(reviews)

def main():
    """Main prediction pipeline"""
    print("Fake Review Detection System")
    print("="*30)
    
    # Load model
    predictor = FakeReviewPredictor()
    
    # Load test data
    test_data = load_sample_data("data/raw/review-Mississippi_10.json", max_reviews=1000)
    
    # Make predictions
    results, features = predictor.predict(test_data)
    
    # Analyze results
    suspicious_reviews = predictor.analyze_suspicious_patterns(results)
    
    # Show top feature importance
    print(f"\nTop 5 Most Important Features:")
    for _, row in predictor.feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save results
    results.to_csv('fake_review_predictions.csv', index=False)
    print(f"\nPredictions saved to: fake_review_predictions.csv")
    
    # Show sample suspicious reviews
    if len(suspicious_reviews) > 0:
        print(f"\nSample Suspicious Reviews:")
        sample_suspicious = suspicious_reviews.nlargest(3, 'fake_probability')[['name', 'rating', 'text', 'fake_probability']]
        for _, review in sample_suspicious.iterrows():
            print(f"\nUser: {review['name']}")
            print(f"Rating: {review['rating']} stars")
            print(f"Text: {str(review['text'])[:100]}...")
            print(f"Fake Probability: {review['fake_probability']:.2%}")

if __name__ == "__main__":
    main()
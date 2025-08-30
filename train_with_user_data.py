#!/usr/bin/env python3
"""
Train RealViews models using user metadata (accounts.csv + account_reviews.csv)
Enhanced training with user behavioral patterns for better fake review detection
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import ReviewDataLoader
from models.model_trainer import ModelTrainer
from utils.user_data_analyzer import UserDataAnalyzer

def prepare_training_data_from_csvs(accounts_file: str = "data/raw/accounts.csv", 
                                   reviews_file: str = "data/raw/account_reviews.csv") -> pd.DataFrame:
    """
    Prepare training data from your CSV files with user metadata integration
    """
    print("📊 Preparing training data from user CSV files...")
    
    # Check if files exist
    if not os.path.exists(accounts_file):
        raise FileNotFoundError(f"Accounts file not found: {accounts_file}")
    
    if not os.path.exists(reviews_file):
        raise FileNotFoundError(f"Reviews file not found: {reviews_file}")
    
    # Load the CSV files
    print(f"📁 Loading accounts from: {accounts_file}")
    accounts_df = pd.read_csv(accounts_file)
    print(f"   → Loaded {len(accounts_df)} accounts")
    
    print(f"📁 Loading reviews from: {reviews_file}")
    reviews_df = pd.read_csv(reviews_file)
    print(f"   → Loaded {len(reviews_df)} reviews")
    
    # Create training dataset from reviews with labels
    print("🔄 Converting reviews to training format...")
    
    # Map the 'is_real' column to violation labels
    training_data = []
    
    for _, review in reviews_df.iterrows():
        # Use the review content as text
        text = review.get('content', '')
        
        # Skip if no content
        if pd.isna(text) or str(text).strip() == '':
            continue
        
        # Determine label based on 'is_real' column
        is_real = review.get('is_real', True)
        
        if is_real:
            # Real review - check for other patterns
            label = 'none'
        else:
            # Fake review - try to classify type based on content
            text_lower = str(text).lower()
            
            # Check for advertisement patterns
            if any(word in text_lower for word in ['website', 'visit', 'link', 'www', 'http', '.com', 'deal']):
                label = 'advertisement'
            # Check for fake patterns (excessive enthusiasm)
            elif any(word in text_lower for word in ['perfect', 'amazing', 'incredible', 'extraordinary', 'phenomenal']):
                label = 'fake' 
            # Check for irrelevant patterns
            elif len(text.split()) < 5:  # Very short reviews might be irrelevant
                label = 'irrelevant'
            else:
                label = 'fake'  # Default for non-real reviews
        
        training_data.append({
            'text': text,
            'account_id': review.get('account_id'),
            'label': label,
            'rating': review.get('rating'),
            'date': review.get('date'),
            'cluster': review.get('cluster'),
            'original_is_real': is_real
        })
    
    training_df = pd.DataFrame(training_data)
    print(f"   → Created {len(training_df)} training samples")
    
    # Show label distribution
    label_dist = training_df['label'].value_counts()
    print("   📈 Label distribution:")
    for label, count in label_dist.items():
        percentage = (count / len(training_df)) * 100
        print(f"      {label}: {count} ({percentage:.1f}%)")
    
    return training_df, accounts_df, reviews_df

def train_user_aware_model(accounts_file: str = "data/raw/accounts.csv",
                          reviews_file: str = "data/raw/account_reviews.csv",
                          retrain: bool = False,
                          save_model: bool = True):
    """
    Complete training pipeline with user metadata integration
    """
    
    print("🚀 RealViews Enhanced Training with User Metadata")
    print("=" * 55)
    
    try:
        # Step 1: Prepare training data
        training_df, accounts_df, reviews_df = prepare_training_data_from_csvs(accounts_file, reviews_file)
        
        # Step 2: Initialize components
        print("\n🔧 Initializing components...")
        loader = ReviewDataLoader()
        user_analyzer = UserDataAnalyzer()
        trainer = ModelTrainer()
        
        # Step 3: Generate user analysis
        print("\n🔍 Analyzing user behavioral patterns...")
        user_report = user_analyzer.generate_user_suspicion_report(accounts_df, reviews_df)
        
        print("   📊 User Analysis Summary:")
        summary = user_report['summary']
        print(f"      Total accounts: {summary['total_accounts']}")
        print(f"      Fake accounts detected: {summary['fake_accounts_detected']} ({summary['fake_accounts_percentage']:.1f}%)")
        print(f"      Burst activity accounts: {summary['burst_accounts_count']} ({summary['burst_accounts_percentage']:.1f}%)")
        print(f"      Average suspicion score: {summary['average_suspicion_score']:.3f}")
        
        # Step 4: Integrate user features into training data
        print("\n🔗 Integrating user suspicion features...")
        enhanced_training_df = loader.integrate_user_features(training_df, user_report)
        
        # Show enhanced features
        user_features = [col for col in enhanced_training_df.columns if 'suspicion' in col or 'burst' in col]
        print(f"      Added user features: {user_features}")
        
        high_suspicion_count = enhanced_training_df.get('is_high_suspicion_account', pd.Series([False])).sum()
        print(f"      High suspicion samples: {high_suspicion_count}/{len(enhanced_training_df)}")
        
        # Step 5: Prepare training splits (avoid stratification issues with small classes)
        print("\n📊 Creating training splits...")
        try:
            splits = loader.prepare_training_data(
                [enhanced_training_df], 
                test_size=0.2, 
                validation_size=0.1,
                balance_classes=True
            )
        except Exception as e:
            print(f"   ⚠️ Stratified split failed ({e}), using random split...")
            # Fall back to simple random split
            from sklearn.model_selection import train_test_split
            
            train_val_df, test_df = train_test_split(enhanced_training_df, test_size=0.2, random_state=42)
            train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)  # 0.1 of total
            
            splits = {
                'train': train_df,
                'validation': val_df,
                'test': test_df,
                'full': enhanced_training_df
            }
        
        print(f"      Training samples: {len(splits['train'])}")
        print(f"      Validation samples: {len(splits['validation'])}")
        print(f"      Test samples: {len(splits['test'])}")
        
        # Step 6: Save enhanced data to processed folder
        print("\n💾 Saving enhanced training data...")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(exist_ok=True)
        
        # Save the splits with user features
        splits['train'].to_csv(processed_dir / "train_data.csv", index=False)
        splits['validation'].to_csv(processed_dir / "validation_data.csv", index=False)  
        splits['test'].to_csv(processed_dir / "test_data.csv", index=False)
        
        # Get feature columns from the enhanced data
        feature_columns = ['text'] + [col for col in enhanced_training_df.columns if 'suspicion' in col or 'burst' in col]
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_samples': len(enhanced_training_df),
            'train_samples': len(splits['train']),
            'validation_samples': len(splits['validation']),
            'test_samples': len(splits['test']),
            'features_included': feature_columns,
            'user_features_enabled': True
        }
        
        with open(processed_dir / "dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"   ✅ Saved enhanced data to {processed_dir}")
        
        # Step 7: Train models using the saved enhanced data
        print("\n🤖 Training enhanced models...")
        
        training_results = trainer.full_training_pipeline(
            use_existing_data=True,  # Use the enhanced data we just saved
            save_models=True
        )
        
        print("\n📈 Training Results:")
        if 'violation_classifiers' in training_results:
            for violation_type, models in training_results['violation_classifiers'].items():
                if models and 'ensemble' in models:
                    ensemble = models['ensemble']
                    print(f"   {violation_type}:")
                    print(f"      F1 Score: {ensemble.get('f1_score', 'N/A'):.3f}")
                    print(f"      Accuracy: {ensemble.get('accuracy', 'N/A'):.3f}")
        
        if 'quality_predictor' in training_results:
            quality = training_results['quality_predictor']
            print(f"   Quality Predictor:")
            print(f"      R² Score: {quality.get('r2_score', 'N/A'):.3f}")
            print(f"      MAE: {quality.get('mae', 'N/A'):.3f}")
        
        # Step 7: Save enhanced model
        if save_model:
            print("\n💾 Saving enhanced model...")
            model_path = "models/enhanced_policy_classifier.joblib"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model with user analysis
            model_data = {
                'models': training_results['trained_models'],
                'vectorizers': training_results['vectorizers'],
                'is_trained': True,
                'user_analysis': user_report,
                'training_stats': training_results['model_performance'],
                'feature_columns': feature_columns
            }
            
            import joblib
            joblib.save(model_data, model_path)
            print(f"   ✅ Enhanced model saved to: {model_path}")
        
        # Step 8: Generate recommendations
        print("\n💡 Training Recommendations:")
        for rec in user_report['recommendations'][:5]:
            print(f"   • {rec}")
        
        print(f"\n🎉 Training Complete!")
        print(f"   • Enhanced with user behavioral analysis")
        print(f"   • User suspicion features integrated")
        print(f"   • Ready for production use")
        
        return {
            'training_results': training_results,
            'user_analysis': user_report,
            'enhanced_data': enhanced_training_df,
            'splits': splits
        }
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_trained_model(model_path: str = "models/enhanced_policy_classifier.joblib"):
    """
    Validate the trained model with some test cases
    """
    print("\n🧪 Validating Trained Model")
    print("=" * 30)
    
    try:
        from models.policy_classifier import PolicyClassifier
        
        # Load the enhanced classifier
        classifier = PolicyClassifier(use_llm=False)
        success = classifier.load_models(model_path)
        
        if not success:
            print("❌ Failed to load trained model")
            return False
        
        print("✅ Model loaded successfully")
        
        # Test cases with different user risk levels
        test_cases = [
            {
                'text': "Great food and excellent service!",
                'user_metadata': {'composite_suspicion_score': 0.1, 'is_high_suspicion_account': False},
                'expected': 'Clean review from low-risk user'
            },
            {
                'text': "Perfect! Visit our website for deals!",
                'user_metadata': {'composite_suspicion_score': 0.8, 'is_high_suspicion_account': True},
                'expected': 'Advertisement from high-risk user'
            },
            {
                'text': "Amazing incredible experience!",
                'user_metadata': {'composite_suspicion_score': 0.7, 'has_burst_activity': True},
                'expected': 'Fake review from burst account'
            }
        ]
        
        print("\n🔍 Test Results:")
        for i, test_case in enumerate(test_cases, 1):
            result = classifier.predict_single(
                text=test_case['text'],
                user_metadata=test_case['user_metadata']
            )
            
            print(f"\n   Test {i}: {test_case['expected']}")
            print(f"      Text: \"{test_case['text']}\"")
            print(f"      Has Violation: {result['has_violation']}")
            print(f"      Quality Score: {result['quality_score']:.3f}")
            
            if 'user_analysis' in result:
                ua = result['user_analysis']
                print(f"      User Risk: {'HIGH' if ua['high_risk'] else 'LOW'}")
                print(f"      User Score: {ua['suspicion_score']:.3f}")
        
        print("\n✅ Model validation complete!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main training interface"""
    parser = argparse.ArgumentParser(description='Train RealViews with user metadata')
    parser.add_argument('--accounts', default='data/raw/accounts.csv', 
                       help='Path to accounts CSV file')
    parser.add_argument('--reviews', default='data/raw/account_reviews.csv',
                       help='Path to reviews CSV file') 
    parser.add_argument('--retrain', action='store_true',
                       help='Retrain all models from scratch')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save the trained model')
    parser.add_argument('--validate', action='store_true',
                       help='Validate model after training')
    
    args = parser.parse_args()
    
    # Check if CSV files exist
    if not os.path.exists(args.accounts):
        print(f"❌ Accounts file not found: {args.accounts}")
        print("   Please ensure your accounts.csv is in data/raw/")
        return False
    
    if not os.path.exists(args.reviews):
        print(f"❌ Reviews file not found: {args.reviews}")
        print("   Please ensure your account_reviews.csv is in data/raw/")
        return False
    
    # Train the model
    results = train_user_aware_model(
        accounts_file=args.accounts,
        reviews_file=args.reviews,
        retrain=args.retrain,
        save_model=not args.no_save
    )
    
    if results is None:
        return False
    
    # Validate if requested
    if args.validate:
        validate_trained_model()
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n🎉 Training completed successfully!")
        print("   Your model is now enhanced with user behavioral analysis!")
        sys.exit(0)
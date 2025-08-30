#!/usr/bin/env python3
"""
Train RealViews models with Chinese ASAP dataset integration
Combines existing English/multilingual data with Chinese restaurant reviews
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
from utils.chinese_data_loader import ASAPChineseLoader, load_chinese_training_data

def prepare_multilingual_training_data(include_chinese: bool = True,
                                     include_user_data: bool = True,
                                     balance_languages: bool = True) -> pd.DataFrame:
    """
    Prepare comprehensive multilingual training dataset
    """
    print("🌍 Preparing multilingual training dataset...")
    
    datasets = []
    
    # 1. Load existing user data (English + some multilingual)
    if include_user_data:
        print("📊 Loading existing user review data...")
        try:
            from train_with_user_data import prepare_training_data_from_csvs
            user_data = prepare_training_data_from_csvs()
            user_data['language'] = 'english'
            user_data['source'] = 'user_data'
            datasets.append(user_data)
            print(f"   → Loaded {len(user_data)} user reviews")
        except Exception as e:
            print(f"   ⚠️  Could not load user data: {e}")
    
    # 2. Load Chinese ASAP dataset
    if include_chinese:
        print("🇨🇳 Loading Chinese ASAP restaurant reviews...")
        try:
            chinese_data = load_chinese_training_data("data/raw/dev.csv")
            if chinese_data:
                # Combine Chinese splits into one dataset
                chinese_combined = pd.concat([
                    chinese_data['train'],
                    chinese_data['validation'], 
                    chinese_data['test']
                ], ignore_index=True)
                chinese_combined['language'] = 'chinese'
                chinese_combined['source'] = 'asap'
                datasets.append(chinese_combined)
                print(f"   → Loaded {len(chinese_combined)} Chinese restaurant reviews")
            else:
                print("   ⚠️  Could not load Chinese data")
        except Exception as e:
            print(f"   ⚠️  Error loading Chinese data: {e}")
    
    if not datasets:
        raise ValueError("No datasets could be loaded!")
    
    # 3. Combine all datasets
    print("🔄 Combining datasets...")
    combined_data = pd.concat(datasets, ignore_index=True)
    
    # Ensure consistent column names
    if 'review_text' in combined_data.columns and 'text' not in combined_data.columns:
        combined_data['text'] = combined_data['review_text']
    elif 'text' in combined_data.columns and 'review_text' not in combined_data.columns:
        combined_data['review_text'] = combined_data['text']
    
    print(f"   → Combined dataset size: {len(combined_data)} samples")
    
    # 4. Language and source distribution
    print("📈 Dataset composition:")
    if 'language' in combined_data.columns:
        lang_dist = combined_data['language'].value_counts()
        for lang, count in lang_dist.items():
            print(f"   {lang}: {count} ({count/len(combined_data)*100:.1f}%)")
    
    if 'source' in combined_data.columns:
        source_dist = combined_data['source'].value_counts()
        for source, count in source_dist.items():
            print(f"   {source}: {count} samples")
    
    # 5. Violation type distribution
    if 'violation_type' in combined_data.columns:
        violation_dist = combined_data['violation_type'].value_counts()
        print("🎯 Violation type distribution:")
        for vtype, count in violation_dist.items():
            print(f"   {vtype}: {count} ({count/len(combined_data)*100:.1f}%)")
    
    return combined_data

def train_multilingual_model(retrain: bool = False, save_model: bool = True):
    """Train model with multilingual data including Chinese"""
    
    print("🚀 RealViews Multilingual Training with Chinese Dataset")
    print("=" * 60)
    
    try:
        # Step 1: Prepare multilingual training data
        training_data = prepare_multilingual_training_data(
            include_chinese=True,
            include_user_data=True,
            balance_languages=True
        )
        
        # Step 2: Initialize components
        print("\n🔧 Initializing training components...")
        loader = ReviewDataLoader()
        trainer = ModelTrainer()
        
        # Step 3: Process data through standard pipeline
        print("\n📊 Processing through RealViews data pipeline...")
        
        # Convert to standard format
        if 'label' not in training_data.columns and 'violation_type' in training_data.columns:
            training_data['label'] = training_data['violation_type']
        
        # Create balanced splits
        splits = loader.prepare_training_data([training_data], balance_classes=True)
        
        print(f"      Training samples: {len(splits['train'])}")
        print(f"      Validation samples: {len(splits['validation'])}")
        print(f"      Test samples: {len(splits['test'])}")
        
        # Step 4: Save enhanced data to processed folder
        print("\n💾 Saving multilingual training data...")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(exist_ok=True)
        
        # Save the splits
        splits['train'].to_csv(processed_dir / "train_data.csv", index=False)
        splits['validation'].to_csv(processed_dir / "validation_data.csv", index=False)
        splits['test'].to_csv(processed_dir / "test_data.csv", index=False)
        
        # Save metadata
        feature_columns = ['text'] + [col for col in training_data.columns 
                                    if col not in ['text', 'review_text', 'label', 'violation_type']]
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_samples': len(training_data),
            'train_samples': len(splits['train']),
            'validation_samples': len(splits['validation']),
            'test_samples': len(splits['test']),
            'languages': ['english', 'chinese'],
            'sources': ['user_data', 'asap'],
            'features_included': feature_columns,
            'multilingual_training': True,
            'chinese_data_included': True
        }
        
        with open(processed_dir / "dataset_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved multilingual data to {processed_dir}")
        
        # Step 5: Train models using the saved multilingual data
        print("\n🤖 Training multilingual models...")
        
        training_results = trainer.full_training_pipeline(
            use_existing_data=True,  # Use the multilingual data we just saved
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
        
        # Step 6: Save enhanced model metadata
        if save_model and training_results:
            print("\n💾 Enhanced multilingual model training completed!")
            print(f"🌍 Model now supports: English + Chinese restaurant reviews")
            print(f"📊 Training data: {len(training_data)} samples across multiple languages")
            
            return training_results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_chinese_analysis():
    """Test the trained model with Chinese reviews"""
    print("\n🧪 Testing Chinese review analysis...")
    
    try:
        # Load the trained model
        from models.policy_classifier import PolicyClassifier
        classifier = PolicyClassifier()
        
        # Test Chinese reviews
        test_reviews = [
            "这家餐厅的食物很好吃，服务也很棒！强烈推荐给大家。",  # Good review
            "菜品质量很差，服务态度也不好，不推荐。",                    # Bad review  
            "优惠活动很多！关注我们微信号获得更多折扣信息！",              # Advertisement
            "这家店位置很好找，交通便利，停车方便。",                    # Location focused
        ]
        
        for i, review in enumerate(test_reviews, 1):
            print(f"\n📝 Test Review {i}: {review}")
            
            # Test both analysis modes
            result_translated = classifier.predict_single(review, analyze_original=False)
            result_original = classifier.predict_single(review, analyze_original=True)
            
            print(f"   🔄 Analyzed via translation: {result_translated['has_violation']} (confidence: {result_translated['confidence']:.3f})")
            print(f"   🇨🇳 Analyzed original Chinese: {result_original['has_violation']} (confidence: {result_original['confidence']:.3f})")
            
            if result_translated['has_violation']:
                violations = [k for k, v in result_translated['violations'].items() if v['detected']]
                print(f"      Violations (translated): {violations}")
            
            if result_original['has_violation']:
                violations = [k for k, v in result_original['violations'].items() if v['detected']]
                print(f"      Violations (original): {violations}")
    
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train RealViews with Chinese data')
    parser.add_argument('--retrain', action='store_true', help='Retrain models from scratch')
    parser.add_argument('--test-only', action='store_true', help='Only test Chinese analysis')
    parser.add_argument('--no-chinese', action='store_true', help='Exclude Chinese dataset')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save models')
    
    args = parser.parse_args()
    
    if args.test_only:
        test_chinese_analysis()
    else:
        # Train model
        results = train_multilingual_model(
            retrain=args.retrain,
            save_model=not args.no_save
        )
        
        if results:
            print("\n✅ Multilingual training completed successfully!")
            print("🎯 Your model now handles Chinese restaurant reviews natively!")
            
            # Test the trained model
            test_chinese_analysis()
        else:
            print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
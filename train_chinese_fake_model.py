#!/usr/bin/env python3
"""
Train RealViews models with Chinese fake review dataset
Integrates chinese_fake.csv into the existing English training pipeline
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
from utils.chinese_fake_loader import load_chinese_fake_training_data, analyze_chinese_fake_dataset

def prepare_multilingual_training_data(include_chinese_fake: bool = True,
                                     include_english: bool = True) -> pd.DataFrame:
    """
    Prepare training dataset with Chinese fake reviews and English data
    """
    print("🌍 Preparing multilingual training dataset with Chinese fake reviews...")
    
    datasets = []
    
    # 1. Load Chinese fake review data
    if include_chinese_fake:
        print("🇨🇳 Loading Chinese fake review dataset...")
        try:
            chinese_splits = load_chinese_fake_training_data("data/raw/chinese_fake.csv")
            if chinese_splits:
                # Combine Chinese splits into one dataset
                chinese_combined = pd.concat([
                    chinese_splits['train'],
                    chinese_splits['validation'], 
                    chinese_splits['test']
                ], ignore_index=True)
                chinese_combined['language'] = 'chinese'
                chinese_combined['source'] = 'chinese_fake'
                datasets.append(chinese_combined)
                print(f"   → Loaded {len(chinese_combined)} Chinese fake/real reviews")
                
                # Show distribution
                fake_count = len(chinese_combined[chinese_combined['violation_type'] == 'fake'])
                real_count = len(chinese_combined[chinese_combined['violation_type'] == 'none'])
                print(f"   → Fake: {fake_count}, Real: {real_count}")
            else:
                print("   ⚠️  Could not load Chinese fake data")
        except Exception as e:
            print(f"   ⚠️  Error loading Chinese fake data: {e}")
    
    # 2. Load existing English training data if available
    if include_english:
        print("🇺🇸 Loading existing English training data...")
        try:
            # Try to load existing processed data
            processed_dir = Path("data/processed")
            if (processed_dir / "train_data.csv").exists():
                english_train = pd.read_csv(processed_dir / "train_data.csv")
                english_val = pd.read_csv(processed_dir / "validation_data.csv")
                english_test = pd.read_csv(processed_dir / "test_data.csv")
                
                english_combined = pd.concat([english_train, english_val, english_test], ignore_index=True)
                english_combined['language'] = english_combined.get('language', 'english')
                english_combined['source'] = english_combined.get('source', 'existing_english')
                datasets.append(english_combined)
                print(f"   → Loaded {len(english_combined)} existing English samples")
            else:
                print("   ⚠️  No existing English training data found")
        except Exception as e:
            print(f"   ⚠️  Error loading English data: {e}")
    
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
    
    # 4. Dataset composition analysis
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

def train_chinese_enhanced_model(retrain: bool = False, save_model: bool = True):
    """Train model with Chinese fake reviews + English data"""
    
    print("🚀 RealViews Training with Chinese Fake Review Dataset")
    print("=" * 60)
    
    try:
        # Step 1: Prepare multilingual training data
        training_data = prepare_multilingual_training_data(
            include_chinese_fake=True,
            include_english=True
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
        print("\n💾 Saving Chinese-enhanced training data...")
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
            'sources': ['chinese_fake', 'existing_english'],
            'features_included': feature_columns,
            'multilingual_training': True,
            'chinese_fake_data_included': True,
            'dataset_info': {
                'chinese_fake_samples': len(training_data[training_data.get('source') == 'chinese_fake']),
                'english_samples': len(training_data[training_data.get('language') == 'english']),
                'fake_reviews': len(training_data[training_data.get('violation_type') == 'fake']),
                'clean_reviews': len(training_data[training_data.get('violation_type') == 'none'])
            }
        }
        
        with open(processed_dir / "dataset_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved Chinese-enhanced data to {processed_dir}")
        
        # Step 5: Train models using the saved multilingual data
        print("\n🤖 Training Chinese-enhanced models...")
        
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
            print("\n💾 Chinese-enhanced model training completed!")
            print(f"🇨🇳 Model now supports: Chinese fake review detection")
            print(f"🌍 Model also supports: English + multilingual via translation")
            print(f"📊 Training data: {len(training_data)} samples across languages")
            
            return training_results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_chinese_fake_analysis():
    """Test the trained model with Chinese fake reviews"""
    print("\n🧪 Testing Chinese fake review detection...")
    
    try:
        # Load the trained model
        from models.policy_classifier import PolicyClassifier
        classifier = PolicyClassifier()
        
        # Test Chinese reviews - both fake and real patterns
        test_reviews = [
            # Potentially fake reviews
            "这个产品非常非常非常好，超级棒！推荐推荐！",  # Excessive repetition
            "宝贝收到了，质量很好很好很好，卖家服务态度很棒很棒！", # Repeated phrases
            "物流很快，包装精美，产品质量超出预期，五星好评！", # Generic positive
            
            # Potentially real reviews  
            "用了一周，整体感觉不错。音质比之前的耳机有提升，但是续航稍微短了点。", # Balanced review
            "外观设计很漂亮，但是实际使用中发现充电速度比宣传的慢一些。", # Mixed sentiment
            "价格相比同类产品有优势，推荐给预算有限的朋友。", # Specific comparison
        ]
        
        for i, review in enumerate(test_reviews, 1):
            print(f"\n📝 Test Review {i}: {review}")
            
            # Test both analysis modes
            result_translated = classifier.predict_single(review, analyze_original=False)
            result_original = classifier.predict_single(review, analyze_original=True)
            
            print(f"   🔄 Via translation: {'VIOLATION' if result_translated['has_violation'] else 'CLEAN'} (confidence: {result_translated['confidence']:.3f})")
            print(f"   🇨🇳 Original Chinese: {'VIOLATION' if result_original['has_violation'] else 'CLEAN'} (confidence: {result_original['confidence']:.3f})")
            
            if result_translated['has_violation']:
                violations = [k for k, v in result_translated['violations'].items() if v['detected']]
                print(f"      Violations (translated): {violations}")
            
            if result_original['has_violation']:
                violations = [k for k, v in result_original['violations'].items() if v['detected']]
                print(f"      Violations (original): {violations}")
    
    except Exception as e:
        print(f"❌ Testing failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train RealViews with Chinese fake data')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze the dataset')
    parser.add_argument('--retrain', action='store_true', help='Retrain models from scratch')
    parser.add_argument('--test-only', action='store_true', help='Only test Chinese analysis')
    parser.add_argument('--no-english', action='store_true', help='Exclude English dataset')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save models')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_chinese_fake_dataset()
    elif args.test_only:
        test_chinese_fake_analysis()
    else:
        # Analyze dataset first
        print("📊 Analyzing Chinese fake dataset...")
        analyze_chinese_fake_dataset()
        
        # Train model
        print("\n" + "="*60)
        results = train_chinese_enhanced_model(
            retrain=args.retrain,
            save_model=not args.no_save
        )
        
        if results:
            print("\n✅ Chinese-enhanced training completed successfully!")
            print("🎯 Your model now handles Chinese fake review detection!")
            
            # Test the trained model
            test_chinese_fake_analysis()
        else:
            print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
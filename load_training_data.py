#!/usr/bin/env python3
"""
RealViews Training Data Loader
Easy script to load your own training data and retrain models

Usage:
    python load_training_data.py                    # Use existing data
    python load_training_data.py --retrain          # Load data and retrain models
    python load_training_data.py --create-examples  # Create example datasets
    python load_training_data.py --validate         # Just validate existing data
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import ReviewDataLoader
from models.model_trainer import ModelTrainer

def create_example_datasets():
    """Create example datasets to show expected format"""
    print("📁 Creating example datasets...")
    
    loader = ReviewDataLoader()
    loader.load_example_datasets()
    
    print("\n✅ Example datasets created in data/raw/")
    print("📋 Available formats:")
    print("   • google_reviews_sample.csv - Google Reviews format")
    print("   • yelp_reviews_sample.json - Yelp format (JSON)")  
    print("   • amazon_reviews_sample.csv - Amazon format")
    print("\n💡 Edit these files or add your own data files to data/raw/")
    
    # Show expected CSV format
    print("\n📝 Expected CSV format:")
    print("   review_text,rating,label")
    print('   "Great food and service!",5,"none"')
    print('   "Visit our website for deals!",5,"advertisement"')
    print('   "Perfect perfect perfect!",5,"fake"')
    print('   "Wrong restaurant entirely.",1,"irrelevant"')

def validate_data():
    """Validate existing training data"""
    print("🔍 Validating training data...")
    
    loader = ReviewDataLoader()
    
    # Load all data files
    csv_datasets = loader.load_csv_files()
    json_datasets = loader.load_json_files()
    all_datasets = csv_datasets + json_datasets
    
    if not all_datasets:
        print("❌ No data files found in data/raw/")
        print("💡 Run with --create-examples to create sample data")
        return False
    
    print(f"📊 Found {len(all_datasets)} datasets")
    
    total_samples = 0
    all_valid = True
    
    # Validate each dataset
    for i, df in enumerate(all_datasets):
        source_file = df.get('source_file', [f'dataset_{i}']).iloc[0] if 'source_file' in df.columns else f'dataset_{i}'
        
        # Standardize columns
        df_std = loader.standardize_columns(df)
        df_std = loader.normalize_violation_labels(df_std)
        
        # Validate
        validation = loader.validate_dataset(df_std)
        total_samples += validation['total_rows']
        
        print(f"\n📋 Dataset: {source_file}")
        print(f"   Rows: {validation['total_rows']}")
        
        if 'label_distribution' in validation:
            print(f"   Labels: {validation['label_distribution']}")
        
        if validation['recommendations']:
            print("   ⚠️  Issues:")
            for rec in validation['recommendations'][:3]:
                print(f"     • {rec}")
            all_valid = False
        else:
            print("   ✅ No issues found")
    
    print(f"\n📊 Total samples across all datasets: {total_samples}")
    
    if all_valid:
        print("✅ All datasets look good!")
    else:
        print("⚠️  Some issues found, but training can still proceed")
    
    return True

def load_and_prepare_data():
    """Load and prepare training data"""
    print("📚 Loading and preparing training data...")
    
    loader = ReviewDataLoader()
    
    # Load data
    csv_datasets = loader.load_csv_files()
    json_datasets = loader.load_json_files()
    all_datasets = csv_datasets + json_datasets
    
    if not all_datasets:
        print("❌ No training data found!")
        print("💡 Add your data files to data/raw/ or run --create-examples")
        return False
    
    # Standardize datasets
    standardized_datasets = []
    for df in all_datasets:
        df_std = loader.standardize_columns(df)
        df_std = loader.normalize_violation_labels(df_std)
        standardized_datasets.append(df_std)
    
    # Create training splits
    splits = loader.prepare_training_data(standardized_datasets, balance_classes=True)
    
    # Save processed data
    loader.save_processed_data(splits)
    
    print(f"✅ Data prepared and saved to data/processed/")
    print(f"   Train: {len(splits['train'])} samples")
    print(f"   Validation: {len(splits['validation'])} samples")
    print(f"   Test: {len(splits['test'])} samples")
    
    return True

def retrain_models():
    """Retrain models with new data"""
    print("🤖 Retraining models with new data...")
    
    trainer = ModelTrainer()
    
    # Run full training pipeline
    results = trainer.full_training_pipeline(
        use_existing_data=True,  # Use the processed data we just created
        save_models=True
    )
    
    if not results:
        print("❌ Training failed!")
        return False
    
    print("\n🎉 Training completed successfully!")
    
    # Show results summary
    if 'violation_classifiers' in results:
        print("\n📊 Violation Classifier Results:")
        for violation_type, models in results['violation_classifiers'].items():
            if models:
                best_f1 = max(model['f1_score'] for model in models.values() if 'f1_score' in model)
                print(f"   {violation_type.title()}: F1 Score = {best_f1:.3f}")
    
    if 'quality_predictor' in results:
        r2 = results['quality_predictor']['r2_score']
        print(f"   Quality Prediction: R² Score = {r2:.3f}")
    
    if 'test_evaluation' in results:
        print("\n📋 Test Set Performance:")
        for name, metrics in results['test_evaluation'].items():
            if 'f1_score' in metrics:
                print(f"   {name}: F1 = {metrics['f1_score']:.3f}")
    
    if 'model_version' in results:
        print(f"\n💾 Models saved as version: {results['model_version']}")
        print(f"   Location: models/saved_models/v_{results['model_version']}/")
    
    print("\n🚀 Your models are now updated! Restart the app to use them.")
    return True

def show_data_format_guide():
    """Show guide for data format requirements"""
    print("""
📋 RealViews Training Data Format Guide

🎯 SUPPORTED FILE FORMATS:
   • CSV files (.csv)
   • JSON files (.json)
   
📂 FILE LOCATION:
   • Place all training files in: data/raw/
   
📊 REQUIRED COLUMNS:
   • text/review_text/content - The review text content
   • label/violation_type - Type of policy violation (optional)
   
🏷️  VIOLATION LABELS:
   • "none" or "clean" - No policy violations
   • "advertisement" - Promotional content
   • "fake" - Fake/synthetic reviews  
   • "irrelevant" - Off-topic content
   
📝 EXAMPLE CSV FORMAT:
   review_text,rating,label
   "Great food and excellent service!",5,"none"
   "Visit our website for 50% off!",5,"advertisement"
   "Perfect perfect perfect amazing!",5,"fake"
   "This is about a different restaurant.",1,"irrelevant"
   
📄 EXAMPLE JSON FORMAT:
   [
     {"text": "Great food!", "stars": 5, "violation": false},
     {"text": "Check our website!", "stars": 5, "violation": true}
   ]
   
💡 TIPS:
   • More data = better models (aim for 100+ examples per category)
   • Balance your classes if possible
   • Include diverse review types and lengths
   • Use real review data when possible
   
🚀 WORKFLOW:
   1. Add your data files to data/raw/
   2. Run: python load_training_data.py --validate
   3. Run: python load_training_data.py --retrain
   4. Restart your RealViews app to use new models
    """)

def main():
    parser = argparse.ArgumentParser(
        description="Load training data and retrain RealViews models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--retrain', action='store_true',
                       help='Load data and retrain models')
    parser.add_argument('--create-examples', action='store_true',
                       help='Create example datasets')
    parser.add_argument('--validate', action='store_true',
                       help='Validate existing training data')
    parser.add_argument('--format-guide', action='store_true',
                       help='Show data format requirements')
    
    args = parser.parse_args()
    
    print("🔍 RealViews Training Data Manager")
    print("=" * 40)
    
    if args.format_guide:
        show_data_format_guide()
        return
    
    if args.create_examples:
        create_example_datasets()
        return
    
    if args.validate:
        validate_data()
        return
    
    if args.retrain:
        print("🚀 Full retraining pipeline starting...")
        
        # Step 1: Validate data
        if not validate_data():
            return
        
        print("\n" + "="*40)
        
        # Step 2: Load and prepare data
        if not load_and_prepare_data():
            return
        
        print("\n" + "="*40)
        
        # Step 3: Retrain models
        if not retrain_models():
            return
        
        print("\n🎉 Complete! Your RealViews models have been updated.")
        
    else:
        # Default: just load and prepare data
        print("📚 Loading and preparing training data...")
        print("💡 Use --retrain to also retrain models after loading")
        
        if validate_data():
            load_and_prepare_data()
        
        print("\n💡 Next steps:")
        print("   • Run with --retrain to update your models")
        print("   • Use --format-guide for data format help")

if __name__ == "__main__":
    main()
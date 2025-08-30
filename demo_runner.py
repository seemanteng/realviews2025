"""
Demo script for RealViews ML Review Filter
Run this to test the system with sample data
"""

import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.policy_classifier import PolicyClassifier
from utils.data_processing import ReviewProcessor

def run_demo(use_llm=False, hf_token=None):
    """Run demonstration with sample data"""
    print("🔍 RealViews Demo - ML-Powered Review Filtering")
    if use_llm:
        print("🤖 LLM-Enhanced Mode Enabled")
    print("=" * 50)
    
    # Load demo data
    try:
        demo_data = pd.read_csv('data/demo_reviews.csv')
        print(f"✅ Loaded {len(demo_data)} demo reviews")
    except FileNotFoundError:
        print("❌ Demo data file not found. Please ensure data/demo_reviews.csv exists.")
        return
    
    # Initialize models
    print(f"\n🤖 Initializing ML models (LLM: {'Enabled' if use_llm else 'Disabled'})...")
    processor = ReviewProcessor()
    classifier = PolicyClassifier(hf_token=hf_token, use_llm=use_llm)
    
    print(f"✅ Models initialized successfully")
    print(f"📊 Model info: {classifier.get_model_info()}")
    
    # Process demo reviews
    print("\n📝 Processing demo reviews...")
    print("-" * 50)
    
    correct_predictions = 0
    total_predictions = len(demo_data)
    
    for idx, row in demo_data.iterrows():
        review_text = row['review_text']
        expected_violation = row['expected_violation']
        expected_type = row['expected_type']
        description = row['description']
        
        # Get prediction
        result = classifier.predict_single(review_text)
        
        # Check accuracy
        prediction_correct = (result['has_violation'] == expected_violation)
        if prediction_correct:
            correct_predictions += 1
        
        # Display results
        status_icon = "✅" if prediction_correct else "❌"
        violation_icon = "🚨" if result['has_violation'] else "✅"
        
        print(f"\n{status_icon} Review {idx + 1}: {description}")
        print(f"   Text: \"{review_text[:60]}{'...' if len(review_text) > 60 else ''}\"")
        print(f"   Expected: {'Violation' if expected_violation else 'Clean'} ({expected_type})")
        print(f"   Predicted: {violation_icon} {'Violation' if result['has_violation'] else 'Clean'}")
        print(f"   Quality Score: {result['quality_score']:.2f} | Confidence: {result['confidence']:.2f}")
        
        if result['has_violation']:
            detected_types = [vtype for vtype, details in result['violations'].items() if details['detected']]
            print(f"   Detected Types: {', '.join(detected_types)}")
        
        if not prediction_correct:
            print(f"   ⚠️  Prediction mismatch - Expected: {expected_type}, Got: {'violation' if result['has_violation'] else 'clean'}")
    
    # Summary
    accuracy = (correct_predictions / total_predictions) * 100
    print("\n" + "=" * 50)
    print(f"📊 DEMO RESULTS SUMMARY")
    print(f"   Total Reviews Processed: {total_predictions}")
    print(f"   Correct Predictions: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    # Performance by category
    print(f"\n📈 Performance by Category:")
    for category in ['advertisement', 'irrelevant', 'fake', 'none']:
        category_data = demo_data[demo_data['expected_type'] == category]
        if len(category_data) > 0:
            category_correct = 0
            for idx, row in category_data.iterrows():
                result = classifier.predict_single(row['review_text'])
                expected_has_violation = row['expected_violation']
                if result['has_violation'] == expected_has_violation:
                    if not expected_has_violation:  # Clean review
                        category_correct += 1
                    elif category in [vtype for vtype, details in result['violations'].items() if details['detected']]:
                        category_correct += 1
            
            category_accuracy = (category_correct / len(category_data)) * 100
            print(f"   {category.title()}: {category_correct}/{len(category_data)} ({category_accuracy:.1f}%)")
    
    print(f"\n🎯 Demo completed! Run 'streamlit run app.py' to use the web interface.")

def test_individual_review(use_llm=False, hf_token=None):
    """Test a single review interactively"""
    print(f"\n🔍 Individual Review Tester ({'LLM-Enhanced' if use_llm else 'Traditional ML'})")
    print("-" * 50)
    
    classifier = PolicyClassifier(hf_token=hf_token, use_llm=use_llm)
    
    while True:
        review = input("\nEnter a review to analyze (or 'quit' to exit): ")
        if review.lower() in ['quit', 'exit', 'q']:
            break
        
        if not review.strip():
            print("❌ Please enter a valid review.")
            continue
        
        print("🔄 Analyzing...")
        result = classifier.predict_single(review)
        
        print(f"\n📊 Analysis Results:")
        print(f"   Status: {'🚨 Violation Detected' if result['has_violation'] else '✅ Clean Review'}")
        print(f"   Quality Score: {result['quality_score']:.2f}")
        print(f"   Confidence: {result['confidence']:.2f}")
        
        if result['has_violation']:
            print(f"   Violations:")
            for vtype, details in result['violations'].items():
                if details['detected']:
                    methods = details.get('methods_detected', [])
                    methods_str = f" (detected by: {', '.join(methods)})" if methods else ""
                    print(f"     - {vtype.title()}{methods_str}: {details['reason'][:100]}...")
        
        print(f"   Explanation: {result['explanation']}")

if __name__ == "__main__":
    # Parse command line arguments
    use_llm = False
    hf_token = None
    interactive = False
    
    for arg in sys.argv[1:]:
        if arg == "--interactive":
            interactive = True
        elif arg == "--llm":
            use_llm = True
        elif arg.startswith("--token="):
            hf_token = arg.split("=", 1)[1]
    
    if interactive:
        test_individual_review(use_llm=use_llm, hf_token=hf_token)
    else:
        run_demo(use_llm=use_llm, hf_token=hf_token)
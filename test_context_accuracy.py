#!/usr/bin/env python3
"""
Test script for improved context accuracy
Tests the specific issues mentioned by the user
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.policy_classifier import PolicyClassifier
from utils.data_processing import ReviewProcessor

def test_context_accuracy():
    """Test context accuracy with problematic examples"""
    print("🧪 Testing Context Accuracy Improvements")
    print("=" * 50)
    
    classifier = PolicyClassifier(use_llm=False)  # Test rule-based improvements
    processor = ReviewProcessor()
    
    # Test cases that should be caught as irrelevant
    test_cases = [
        {
            'text': "I love Taylor Swift",
            'context': "Restaurant: Mario's Italian Restaurant", 
            'expected_relevant': False,
            'description': "Celebrity mention in restaurant context"
        },
        {
            'text': "My dog is so cute and fluffy",
            'context': "Restaurant: Luigi's Pizza",
            'expected_relevant': False,
            'description': "Personal pet in restaurant context"
        },
        {
            'text': "The weather is really nice today",
            'context': "Hotel: Hilton Downtown",
            'expected_relevant': False,
            'description': "Weather talk in hotel context"
        },
        {
            'text': "I love playing football with my friends",
            'context': "Product: iPhone 15 Pro",
            'expected_relevant': False,
            'description': "Sports in product context"
        },
        # Valid examples that should pass
        {
            'text': "Great food and excellent service! The pasta was delicious.",
            'context': "Restaurant: Mario's Italian Restaurant",
            'expected_relevant': True,
            'description': "Valid restaurant review"
        },
        {
            'text': "The room was clean and the bed was comfortable.",
            'context': "Hotel: Hilton Downtown",
            'expected_relevant': True,
            'description': "Valid hotel review"
        },
        {
            'text': "Fast delivery and good quality. Works as expected.",
            'context': "Product: iPhone 15 Pro",
            'expected_relevant': True,
            'description': "Valid product review"
        }
    ]
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['description']}")
        print(f"   Text: \"{test_case['text']}\"")
        print(f"   Context: {test_case['context']}")
        
        # Get prediction
        result = classifier.predict_single(test_case['text'], product_info=test_case['context'])
        
        # Check context analysis
        if 'context' in result:
            context_info = result['context']
            relevance_score = context_info['relevance_score']
            
            print(f"   Relevance Score: {relevance_score:.3f}")
            print(f"   Quality Score: {result['quality_score']:.3f}")
            
            # Determine if system considers it relevant (threshold 0.4)
            system_considers_relevant = relevance_score >= 0.4 and not result['has_violation']
            expected = test_case['expected_relevant']
            
            if system_considers_relevant == expected:
                print("   ✅ CORRECT prediction")
                correct_predictions += 1
            else:
                print("   ❌ INCORRECT prediction")
                print(f"      Expected relevant: {expected}")
                print(f"      System says relevant: {system_considers_relevant}")
            
            # Show violations if any
            if result['has_violation']:
                violations = [v for v, details in result['violations'].items() if details['detected']]
                print(f"   Violations: {', '.join(violations)}")
            
            print(f"   Assessment: {context_info.get('overall_assessment', 'N/A')}")
            
        else:
            print("   ⚠️  No context analysis available")
    
    print("\n" + "=" * 50)
    accuracy = (correct_predictions / total_tests) * 100
    print(f"📊 ACCURACY RESULTS:")
    print(f"   Correct: {correct_predictions}/{total_tests}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 85:
        print("🎉 EXCELLENT - Context accuracy is now much improved!")
    elif accuracy >= 70:
        print("👍 GOOD - Context accuracy is better but could be improved further")
    else:
        print("⚠️  NEEDS WORK - Context accuracy still needs improvement")
    
    return accuracy

def test_specific_user_case():
    """Test the specific case mentioned by the user"""
    print("\n🎯 Testing Specific User Case")
    print("=" * 30)
    
    classifier = PolicyClassifier(use_llm=False)
    
    text = "I love Taylor Swift"
    context = "Restaurant: Mario's Italian Restaurant"
    
    print(f"Text: \"{text}\"")
    print(f"Context: {context}")
    
    result = classifier.predict_single(text, product_info=context)
    
    if 'context' in result:
        context_info = result['context']
        relevance = context_info['relevance_score']
        
        print(f"\nResults:")
        print(f"   Relevance Score: {relevance:.3f}")
        print(f"   Quality Score: {result['quality_score']:.3f}")
        print(f"   Has Violation: {result['has_violation']}")
        print(f"   Assessment: {context_info['overall_assessment']}")
        
        if relevance < 0.4 or result['has_violation']:
            print("✅ SUCCESS: Now correctly detects as problematic!")
        else:
            print("❌ STILL FAILING: System still considers this valid")
    else:
        print("No context analysis available")

if __name__ == "__main__":
    print("🔍 RealViews Context Accuracy Test")
    print("Testing improvements to context relevance detection")
    print("\n")
    
    # Test the specific user case first
    test_specific_user_case()
    
    print("\n" + "="*60)
    
    # Run full accuracy test
    accuracy = test_context_accuracy()
    
    print(f"\n🎯 The 'I love Taylor Swift' case should now be correctly flagged!")
    print(f"   Run the web app and test it yourself to confirm.")
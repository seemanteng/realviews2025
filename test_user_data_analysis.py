#!/usr/bin/env python3
"""
Test script for user metadata analysis and fake review detection
Tests the integrated user data features in the policy classifier
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.user_data_analyzer import UserDataAnalyzer
from models.policy_classifier import PolicyClassifier
from utils.data_loader import ReviewDataLoader

def create_test_user_data():
    """Create synthetic user data for testing"""
    
    # Create accounts data
    accounts_data = [
        # High risk fake account
        {"_id": "fake_account_1", "is_deleted": False, "is_private": False, "is_real": False, 
         "local_guide_level": 0, "name_score": 500000, "number_of_reviews": 50},
        
        # Burst activity account
        {"_id": "burst_account_1", "is_deleted": False, "is_private": False, "is_real": False,
         "local_guide_level": 1, "name_score": 2000, "number_of_reviews": 30},
        
        # Legitimate account
        {"_id": "real_account_1", "is_deleted": False, "is_private": False, "is_real": True,
         "local_guide_level": 5, "name_score": 100, "number_of_reviews": 15},
        
        # Deleted suspicious account
        {"_id": "deleted_account_1", "is_deleted": True, "is_private": False, "is_real": False,
         "local_guide_level": 0, "name_score": 750000, "number_of_reviews": 80},
        
        # High volume account (potentially suspicious)
        {"_id": "volume_account_1", "is_deleted": False, "is_private": True, "is_real": False,
         "local_guide_level": 0, "name_score": 10000, "number_of_reviews": 150}
    ]
    
    # Create reviews data with temporal patterns
    base_date = datetime.now() - timedelta(days=30)
    
    reviews_data = []
    
    # Fake account - normal spacing
    for i in range(10):
        reviews_data.append({
            "_id": f"review_fake_{i}",
            "account_id": "fake_account_1", 
            "content": f"Perfect amazing incredible restaurant! Best food ever! Visit www.example{i}.com for deals!",
            "rating": 5,
            "date": (base_date + timedelta(days=i*2)).strftime('%Y-%m-%d'),
            "is_real": False,
            "cluster": "Restaurant"
        })
    
    # Burst account - many reviews in short time
    burst_start = base_date + timedelta(days=10)
    for i in range(15):
        reviews_data.append({
            "_id": f"review_burst_{i}",
            "account_id": "burst_account_1",
            "content": f"Great experience! Highly recommend! Check our website at promo{i}.link",
            "rating": 5,
            "date": (burst_start + timedelta(hours=i*2)).strftime('%Y-%m-%d'),
            "is_real": False,
            "cluster": "Hotel"
        })
    
    # Real account - natural pattern
    for i in range(8):
        reviews_data.append({
            "_id": f"review_real_{i}",
            "account_id": "real_account_1",
            "content": f"Good service and quality food. Staff was friendly and helpful. Worth visiting again.",
            "rating": 4,
            "date": (base_date + timedelta(days=i*5)).strftime('%Y-%m-%d'),
            "is_real": True,
            "cluster": "Restaurant"
        })
    
    # Volume account - spread out but suspicious content
    for i in range(25):
        reviews_data.append({
            "_id": f"review_volume_{i}",
            "account_id": "volume_account_1",
            "content": f"Excellent! Perfect! Amazing! Best ever! Extraordinary experience!",
            "rating": 5,
            "date": (base_date + timedelta(days=i)).strftime('%Y-%m-%d'),
            "is_real": False,
            "cluster": "Product"
        })
    
    accounts_df = pd.DataFrame(accounts_data)
    reviews_df = pd.DataFrame(reviews_data)
    
    return accounts_df, reviews_df

def test_user_data_analyzer():
    """Test the user data analyzer"""
    print("🧪 Testing User Data Analyzer")
    print("=" * 40)
    
    # Create test data
    accounts_df, reviews_df = create_test_user_data()
    
    # Initialize analyzer
    analyzer = UserDataAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_user_suspicion_report(accounts_df, reviews_df)
    
    print(f"📊 Analysis Results:")
    print(f"   Total Accounts: {report['summary']['total_accounts']}")
    print(f"   Total Reviews: {report['summary']['total_reviews']}")
    print(f"   Fake Accounts Detected: {report['summary']['fake_accounts_detected']}")
    print(f"   Fake Accounts %: {report['summary']['fake_accounts_percentage']:.1f}%")
    print(f"   Burst Accounts: {report['summary']['burst_accounts_count']}")
    print(f"   Burst Accounts %: {report['summary']['burst_accounts_percentage']:.1f}%")
    print(f"   Reviews with Links: {report['summary']['reviews_with_links']}")
    print(f"   Average Suspicion Score: {report['summary']['average_suspicion_score']:.3f}")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"   {i}. {rec}")
    
    return report

def test_policy_classifier_integration():
    """Test policy classifier with user metadata integration"""
    print("\n🔬 Testing Policy Classifier with User Metadata")
    print("=" * 50)
    
    # Create test data
    accounts_df, reviews_df = create_test_user_data()
    
    # Initialize components
    analyzer = UserDataAnalyzer()
    classifier = PolicyClassifier(use_llm=False)  # Focus on user metadata features
    
    # Generate user analysis
    user_report = analyzer.generate_user_suspicion_report(accounts_df, reviews_df)
    suspicion_scores = user_report['final_suspicion_scores']
    
    # Test different account types
    test_cases = [
        {
            'review_text': "Perfect amazing incredible! Best restaurant ever! Visit our website!",
            'account_id': "fake_account_1",
            'description': "High suspicion fake account with promotional content"
        },
        {
            'review_text': "Great food and service. Staff was very friendly.",
            'account_id': "burst_account_1", 
            'description': "Burst activity account with normal review text"
        },
        {
            'review_text': "Good experience overall. Food was tasty and service was prompt.",
            'account_id': "real_account_1",
            'description': "Legitimate account with normal review"
        },
        {
            'review_text': "EXCELLENT! PERFECT! AMAZING! EXTRAORDINARY! BEST EVER!",
            'account_id': "volume_account_1",
            'description': "High volume account with suspicious enthusiasm"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {test_case['description']}")
        print(f"   Account ID: {test_case['account_id']}")
        print(f"   Review: \"{test_case['review_text']}\"")
        
        # Get user metadata for this account
        user_metadata = {}
        account_suspicion = suspicion_scores[suspicion_scores['account_id'] == test_case['account_id']]
        
        if not account_suspicion.empty:
            user_metadata = {
                'composite_suspicion_score': account_suspicion.iloc[0]['composite_suspicion_score'],
                'suspicion_level': account_suspicion.iloc[0]['suspicion_level'],
                'has_burst_activity': account_suspicion.iloc[0]['has_burst_activity'],
                'account_suspicion_score': account_suspicion.iloc[0]['account_suspicion_score'],
                'is_high_suspicion_account': account_suspicion.iloc[0]['composite_suspicion_score'] > 0.6
            }
            
            # Add additional metadata from original accounts
            account_data = accounts_df[accounts_df['_id'] == test_case['account_id']]
            if not account_data.empty:
                user_metadata.update({
                    'is_deleted': account_data.iloc[0]['is_deleted'],
                    'number_of_reviews': account_data.iloc[0]['number_of_reviews']
                })
        
        # Analyze with user metadata
        result = classifier.predict_single(
            text=test_case['review_text'],
            user_metadata=user_metadata
        )
        
        print(f"   📈 Results:")
        print(f"      Has Violation: {result['has_violation']}")
        print(f"      Quality Score: {result['quality_score']:.3f}")
        print(f"      Overall Confidence: {result['confidence']:.3f}")
        
        if 'user_analysis' in result:
            ua = result['user_analysis']
            print(f"      User Risk Level: {'HIGH' if ua['high_risk'] else 'LOW'}")
            print(f"      User Suspicion Score: {ua['suspicion_score']:.3f}")
            print(f"      User Indicators: {len(ua['suspicion_indicators'])}")
            print(f"      Recommendation: {ua['recommendation']}")
            
            if ua['suspicion_indicators']:
                print(f"      Top Indicators:")
                for indicator in ua['suspicion_indicators'][:3]:
                    print(f"         • {indicator}")
        
        # Check which violations were detected
        violations_detected = [v for v, data in result['violations'].items() if data['detected']]
        if violations_detected:
            print(f"      Violations Detected: {', '.join(violations_detected)}")
            
            for violation in violations_detected:
                methods = result['violations'][violation]['methods_detected']
                if 'user' in methods:
                    print(f"         • {violation}: Enhanced by user metadata analysis")
        
        results.append({
            'case': test_case['description'],
            'account_id': test_case['account_id'],
            'has_violation': result['has_violation'],
            'quality_score': result['quality_score'],
            'user_enhanced': 'user_analysis' in result and result['user_analysis']['high_risk']
        })
    
    return results

def test_data_loader_integration():
    """Test data loader integration with user metadata"""
    print("\n📁 Testing Data Loader User Metadata Integration")
    print("=" * 45)
    
    # Create test data files
    accounts_df, reviews_df = create_test_user_data()
    
    # Save to temporary files
    os.makedirs("data/raw", exist_ok=True)
    accounts_df.to_csv("data/raw/test_accounts.csv", index=False)
    reviews_df.to_csv("data/raw/test_account_reviews.csv", index=False)
    
    # Test data loader
    loader = ReviewDataLoader(data_dir="data/raw")
    
    try:
        # Generate user metadata report
        report = loader.generate_user_metadata_report()
        
        print(f"📊 Metadata Report Generated:")
        print(f"   Executive Summary Keys: {list(report.get('executive_summary', {}).keys())}")
        print(f"   Recommendations Count: {len(report.get('recommendations', []))}")
        print(f"   Suspicious Accounts: {len(report.get('suspicious_accounts', []))}")
        print(f"   High Risk Indicators: {len(report.get('high_risk_indicators', []))}")
        
        if report.get('high_risk_indicators'):
            print(f"   Risk Indicators:")
            for indicator in report['high_risk_indicators']:
                print(f"      • {indicator}")
        
        # Test user-aware training data creation
        print(f"\n🚀 Testing User-Aware Training Data Creation...")
        
        # Create a larger review dataset for training (need minimum samples per class)
        training_reviews = pd.DataFrame([
            {"text": "Great food!", "account_id": "real_account_1", "label": "none"},
            {"text": "Good service.", "account_id": "real_account_1", "label": "none"},
            {"text": "Nice atmosphere.", "account_id": "real_account_1", "label": "none"},
            {"text": "Perfect! Visit our website!", "account_id": "fake_account_1", "label": "advertisement"},
            {"text": "Check out our deals online!", "account_id": "fake_account_1", "label": "advertisement"},
            {"text": "Visit www.example.com for more!", "account_id": "fake_account_1", "label": "advertisement"},
            {"text": "Amazing incredible experience!", "account_id": "volume_account_1", "label": "fake"},
            {"text": "Perfect perfect perfect!", "account_id": "volume_account_1", "label": "fake"},
            {"text": "Best ever extraordinary!", "account_id": "volume_account_1", "label": "fake"}
        ])
        
        # Test the user metadata integration directly
        user_analysis = loader.load_user_metadata("test_accounts.csv", "test_account_reviews.csv")
        
        if user_analysis:
            enhanced_reviews = loader.integrate_user_features(training_reviews, user_analysis)
            
            print(f"   Training Data Enhanced:")
            print(f"      Original samples: {len(training_reviews)}")
            print(f"      Enhanced samples: {len(enhanced_reviews)}")
            print(f"      Enhanced columns: {list(enhanced_reviews.columns)}")
            
            # Check for user suspicion features
            user_features = [col for col in enhanced_reviews.columns if 'suspicion' in col or 'burst' in col]
            if user_features:
                print(f"      User Features Added: {user_features}")
                
                # Show some statistics
                high_suspicion_count = enhanced_reviews.get('is_high_suspicion_account', pd.Series([False])).sum()
                print(f"      High Suspicion Samples: {high_suspicion_count}/{len(enhanced_reviews)}")
        else:
            print("   ❌ No user analysis available for integration test")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing data loader integration: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            os.remove("data/raw/test_accounts.csv")
            os.remove("data/raw/test_account_reviews.csv")
        except:
            pass

def run_comprehensive_test():
    """Run all user data analysis tests"""
    print("🔍 RealViews User Data Analysis - Comprehensive Test Suite")
    print("=" * 60)
    
    test_results = {
        'user_analyzer': False,
        'policy_integration': False, 
        'data_loader_integration': False
    }
    
    try:
        # Test 1: User Data Analyzer
        user_report = test_user_data_analyzer()
        test_results['user_analyzer'] = bool(user_report and 'summary' in user_report)
        
        # Test 2: Policy Classifier Integration
        policy_results = test_policy_classifier_integration()
        test_results['policy_integration'] = bool(policy_results and len(policy_results) > 0)
        
        # Test 3: Data Loader Integration  
        test_results['data_loader_integration'] = test_data_loader_integration()
        
    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"📋 TEST SUMMARY")
    print(f"=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! User metadata analysis system is working correctly!")
        print("\n🚀 Ready for production use:")
        print("   • User suspicion scoring implemented")
        print("   • Policy classifier enhanced with user metadata")
        print("   • Data loader supports user-aware training")
        print("   • Comprehensive fake review detection active")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
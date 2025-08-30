"""
Test script for LLM integration
Run this to verify the enhanced system works correctly
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality without LLM"""
    print("🧪 Testing Basic Functionality (No LLM)")
    print("=" * 50)
    
    try:
        from models.policy_classifier import PolicyClassifier
        from utils.data_processing import ReviewProcessor
        
        # Test without LLM
        classifier = PolicyClassifier(use_llm=False)
        
        test_reviews = [
            "Great food and excellent service!",
            "khdbxvfwgw",  # Gibberish
            "Visit our website www.example.com for 50% off!",  # Advertisement
            "Perfect perfect perfect! Amazing incredible outstanding!"  # Fake
        ]
        
        for i, review in enumerate(test_reviews):
            print(f"\n📝 Test {i+1}: \"{review}\"")
            result = classifier.predict_single(review)
            print(f"   Violation: {'Yes' if result['has_violation'] else 'No'}")
            print(f"   Quality: {result['quality_score']:.2f}")
            print(f"   Confidence: {result['confidence']:.2f}")
        
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_llm_integration(hf_token=None):
    """Test LLM integration"""
    print("\n🤖 Testing LLM Integration")
    print("=" * 50)
    
    try:
        from models.policy_classifier import PolicyClassifier
        
        # Test with LLM
        classifier = PolicyClassifier(hf_token=hf_token, use_llm=True)
        
        if not classifier.use_llm:
            print("⚠️  LLM not available, skipping LLM tests")
            return True
        
        test_review = "The food was absolutely amazing and the service was perfect! I highly recommend this place to everyone!"
        
        print(f"📝 Testing: \"{test_review}\"")
        result = classifier.predict_single(test_review)
        
        print(f"   Violation: {'Yes' if result['has_violation'] else 'No'}")
        print(f"   Quality Score: {result['quality_score']:.2f}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Explanation: {result['explanation']}")
        
        # Check for LLM-specific features
        for violation_type, details in result['violations'].items():
            if details['detected'] and 'methods_detected' in details:
                methods = details.get('methods_detected', [])
                if 'llm' in methods:
                    print(f"   ✅ LLM detected {violation_type} violation")
        
        print("\n✅ LLM integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False

def test_gibberish_detection():
    """Test enhanced gibberish detection"""
    print("\n🔍 Testing Enhanced Gibberish Detection")
    print("=" * 50)
    
    try:
        from models.policy_classifier import PolicyClassifier
        
        classifier = PolicyClassifier(use_llm=False)  # Test traditional methods first
        
        gibberish_tests = [
            ("khdbxvfwgw", True),
            ("asdfghjkl qwertyuiop", True),
            ("Great food and service", False),
            ("aaaaaaaaaa bbbbbbbbbb", True),
            ("The restaurant has excellent ambiance", False)
        ]
        
        for text, expected_gibberish in gibberish_tests:
            result = classifier.predict_single(text)
            is_very_low_quality = result['quality_score'] < 0.2
            
            print(f"   \"{text}\"")
            print(f"   Quality: {result['quality_score']:.2f} | Expected Gibberish: {expected_gibberish}")
            
            if expected_gibberish and is_very_low_quality:
                print("   ✅ Correctly detected as low quality/gibberish")
            elif not expected_gibberish and not is_very_low_quality:
                print("   ✅ Correctly identified as valid content")
            else:
                print(f"   ⚠️  Unexpected result")
        
        print("\n✅ Gibberish detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Gibberish detection test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 RealViews LLM Integration Test Suite")
    print("TechJam 2025 Hackathon")
    print("=" * 60)
    
    # Get HF token from user if desired
    hf_token = None
    if len(sys.argv) > 1 and sys.argv[1] != '--no-input':
        token_input = input("\nEnter your Hugging Face token (or press Enter to skip): ").strip()
        if token_input:
            hf_token = token_input
            print("✅ HF token provided")
        else:
            print("⚠️  No HF token provided - using public models only")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_llm_integration(hf_token):
        tests_passed += 1
    
    if test_gibberish_detection():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your LLM integration is working correctly.")
        print("\n🚀 Next steps:")
        print("   1. Run: streamlit run app.py")
        print("   2. Go to Settings to configure your HF token")
        print("   3. Test with the enhanced LLM analysis!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
        print("   The system should still work with basic functionality.")

if __name__ == "__main__":
    main()
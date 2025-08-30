"""
Setup script for RealViews ML Review Filter
TechJam 2025 Hackathon Submission
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True) 
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")
        print("   The system will still work with reduced functionality.")
        return False

def verify_installation():
    """Verify that all components are working"""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import streamlit
        import pandas
        import numpy
        import sklearn
        from utils.data_processing import ReviewProcessor
        from models.policy_classifier import PolicyClassifier
        
        print("✅ All core modules imported successfully!")
        
        # Test model initialization
        processor = ReviewProcessor()
        classifier = PolicyClassifier()
        print("✅ ML models initialized successfully!")
        
        # Test a simple prediction
        test_review = "Great food and excellent service!"
        result = classifier.predict_single(test_review)
        print("✅ Prediction system working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        'data/processed',
        'data/raw',
        'assets',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ {directory}")
    
    print("✅ Directories created successfully!")

def main():
    """Main setup function"""
    print("🔍 RealViews Setup - ML-Powered Review Filtering System")
    print("TechJam 2025 Hackathon Submission")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at dependency installation")
        return False
    
    # Download NLTK data
    download_nltk_data()
    
    # Verify installation
    if not verify_installation():
        print("❌ Setup failed at verification")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n🚀 Next steps:")
    print("   1. Run the demo: python demo_runner.py")
    print("   2. Start the web app: streamlit run app.py")
    print("   3. Open browser to: http://localhost:8501")
    print("\n📖 For more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
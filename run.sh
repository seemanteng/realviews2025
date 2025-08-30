#!/bin/bash

# RealViews ML Review Filter - Quick Start Script
# TechJam 2025 Hackathon Submission

echo "🔍 RealViews - ML-Powered Review Filtering System"
echo "TechJam 2025 Hackathon Submission"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "   Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python detected: $(python3 --version)"

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is required but not installed."
    echo "   Please install pip and try again."
    exit 1
fi

# Run setup if requirements.txt exists and packages aren't installed
if [ ! -d "venv" ] && [ -f "requirements.txt" ]; then
    echo ""
    echo "🔧 First-time setup detected. Installing dependencies..."
    echo "   This may take a few minutes..."
    
    # Try to install requirements, fallback to lite version if needed
    if command -v pip3 &> /dev/null; then
        pip3 install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "⚠️  Full requirements failed. Trying lightweight version..."
            pip3 install -r requirements-lite.txt
        fi
    else
        pip install -r requirements.txt
        if [ $? -ne 0 ]; then
            echo "⚠️  Full requirements failed. Trying lightweight version..."
            pip install -r requirements-lite.txt
        fi
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        echo "   Please try: pip install streamlit pandas numpy matplotlib"
        exit 1
    fi
    
    echo "✅ Dependencies installed successfully!"
fi

# Check for demo option
if [ "$1" == "--demo" ] || [ "$1" == "-d" ]; then
    echo ""
    echo "🎬 Running demo with sample data..."
    python3 demo_runner.py
    exit 0
fi

# Check for interactive test option
if [ "$1" == "--test" ] || [ "$1" == "-t" ]; then
    echo ""
    echo "🧪 Running interactive test mode..."
    python3 demo_runner.py --interactive
    exit 0
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    if command -v pip3 &> /dev/null; then
        pip3 install -r requirements.txt
    else
        pip install -r requirements.txt
    fi
fi

# Start the Streamlit application
echo ""
echo "🚀 Starting RealViews web application..."
echo "   Opening in browser at: http://localhost:8501"
echo ""
echo "   Press Ctrl+C to stop the application"
echo "   Use --demo flag to run demo first: ./run.sh --demo"
echo ""

python3 -m streamlit run app.py

echo ""
echo "👋 Thanks for using RealViews!"
echo "   Built for TechJam 2025 Hackathon"
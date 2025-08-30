#!/usr/bin/env python3
"""
Startup script for RealViews on Netlify
"""
import streamlit as st
import subprocess
import sys

def main():
    """Start the Streamlit app"""
    try:
        # Run Streamlit with the main app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except Exception as e:
        st.error(f"Failed to start app: {e}")

if __name__ == "__main__":
    main()
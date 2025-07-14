# quick_test.py
#!/usr/bin/env python3
"""
Quick test script for the full-featured pipeline
"""

import os
import sys
from pathlib import Path

def test_full_pipeline():
    """Test all components of the full pipeline"""
    
    print("🧪 Testing Full-Featured YC Pipeline Components")
    print("=" * 50)
    
    success_count = 0
    total_tests = 6
    
    # Test 1: Environment setup
    print("1. Testing environment setup...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key and api_key != 'your_anthropic_api_key_here':
            print("   ✅ Anthropic API key configured")
            success_count += 1
        else:
            print("   ⚠️  Anthropic API key not configured")
    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
    
    # Test 2: Required packages
    print("2. Testing required packages...")
    try:
        import requests
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from bs4 import BeautifulSoup
        import schedule
        print("   ✅ All required packages available")
        success_count += 1
    except ImportError as e:
        print(f"   

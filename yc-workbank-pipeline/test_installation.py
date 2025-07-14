#!/usr/bin/env python3
"""
Test script to verify installation
"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported"""
    required_packages = [
        'requests',
        'pandas',
        'numpy',
        'bs4',  # beautifulsoup4
    ]
    
    optional_packages = [
        'anthropic',
        'openai',
        'flask',
        'schedule',
        'psutil'
    ]
    
    print("Testing required packages...")
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            return False
    
    print("\nTesting optional packages...")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"⚠️  {package} - WARNING: {e}")
    
    return True

def test_directories():
    """Test that directories exist"""
    import os
    required_dirs = ['data', 'output', 'logs', 'config']
    
    print("\nTesting directories...")
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ - OK")
        else:
            print(f"❌ {directory}/ - MISSING")
            return False
    
    return True

def test_config():
    """Test configuration"""
    import os
    print("\nTesting configuration...")
    
    if os.path.exists('.env'):
        print("✅ .env file - OK")
        
        # Check if API key is set
        from dotenv import load_dotenv
        load_dotenv()
        
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if anthropic_key and anthropic_key != 'your_anthropic_api_key_here':
            print("✅ Anthropic API key - CONFIGURED")
        elif openai_key and openai_key != 'your_openai_api_key_here':
            print("✅ OpenAI API key - CONFIGURED")
        else:
            print("⚠️  API keys - NOT CONFIGURED (you'll need to add these)")
    else:
        print("❌ .env file - MISSING")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing YC-WORKBank Pipeline Installation")
    print("=" * 45)
    
    success = True
    success &= test_imports()
    success &= test_directories()
    success &= test_config()
    
    print("\n" + "=" * 45)
    if success:
        print("🎉 Installation test PASSED!")
        print("\nNext steps:")
        print("1. Add your API key to .env file")
        print("2. Replace sample data with real WORKBank tasks")
        print("3. Download the pipeline Python files")
        print("4. Run: python cli.py run")
    else:
        print("❌ Installation test FAILED!")
        print("Please check the errors above and fix them.")
        sys.exit(1)

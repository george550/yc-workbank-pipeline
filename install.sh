#!/bin/bash

# YC-WORKBank Pipeline Installer
# This script automates the installation process

set -e  # Exit on any error

echo "🚀 YC-WORKBank Pipeline Installer"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_step() {
    echo -e "${BLUE}📋 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Check prerequisites
print_step "Checking prerequisites..."

# Check Python
if check_command python3; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    REQUIRED_VERSION="3.8"
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        print_success "Python version $PYTHON_VERSION is compatible"
    else
        print_error "Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
elif check_command python; then
    PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
        print_success "Python version $PYTHON_VERSION is compatible"
        PYTHON_CMD="python"
    else
        print_error "Python $REQUIRED_VERSION or higher is required"
        exit 1
    fi
else
    print_error "Python is not installed. Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Set Python command
PYTHON_CMD=${PYTHON_CMD:-python3}

# Check pip
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    print_error "pip is not installed. Please install pip first."
    exit 1
fi

PIP_CMD=$(command -v pip3 2>/dev/null || command -v pip)

echo ""
print_step "Setting up project directory..."

# Ask for installation directory
read -p "Enter installation directory (default: ./yc-workbank-pipeline): " INSTALL_DIR
INSTALL_DIR=${INSTALL_DIR:-./yc-workbank-pipeline}

# Create directory
if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
    print_success "Created directory: $INSTALL_DIR"
else
    print_warning "Directory already exists: $INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# Create virtual environment
print_step "Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate
print_success "Virtual environment activated"

# Create requirements.txt
print_step "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core dependencies
requests>=2.31.0
beautifulsoup4>=4.12.0
pandas>=2.0.0
numpy>=1.24.0

# LLM APIs
anthropic>=0.25.0
openai>=1.30.0

# Google Sheets integration (optional)
gspread>=5.12.0
google-auth>=2.23.0
google-auth-oauthlib>=1.0.0
google-auth-httplib2>=0.1.0

# Web interface
flask>=2.3.0

# Data processing
lxml>=4.9.0
openpyxl>=3.1.0

# Utility
python-dotenv>=1.0.0
schedule>=1.2.0
psutil>=5.9.0

# Development (optional)
pytest>=7.0.0
flake8>=6.0.0
black>=23.0.0
EOF

# Install dependencies
print_step "Installing Python dependencies..."
$PIP_CMD install -r requirements.txt
print_success "Dependencies installed"

# Create directory structure
print_step "Creating directory structure..."
mkdir -p data output logs config templates
print_success "Directories created"

# Create .env template
print_step "Creating configuration template..."
cat > .env.template << 'EOF'
# API Keys (choose one)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=anthropic

# Data Files
WORKBANK_FILE=data/workbank_tasks.csv
OUTPUT_DIR=output

# Storage
STORAGE_TYPE=csv
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_SHEET_ID=your_google_sheet_id_here

# Rate Limiting
API_RATE_LIMIT_DELAY=1.0
SCRAPING_DELAY=0.5
BATCH_SIZE=10

# Web Interface
FLASK_SECRET_KEY=your_secret_key_here
EOF

# Copy to .env if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.template .env
    print_success "Created .env file from template"
else
    print_warning ".env file already exists"
fi

# Create sample WORKBank data
print_step "Creating sample WORKBank data..."
cat > data/workbank_tasks.csv << 'EOF'
task_id,occupation,task_description,onet_soc_code
1,Computer Programmers,"Write, update, and maintain computer programs or software packages to handle specific jobs such as tracking inventory, storing or retrieving data, or controlling other equipment.",15-1251.00
2,Customer Service Representatives,"Interact with customers to provide information in response to inquiries about products and services and to handle and resolve complaints.",43-4051.00
3,Financial Analysts,"Conduct quantitative analyses of information involving investment programs or consideration of the purchase or sale of securities.",13-2051.00
4,Software Developers,"Develop, create, and modify general computer applications software or specialized utility programs.",15-1132.00
5,Market Research Analysts,"Research market conditions in local, regional, or national areas, or gather information to determine potential sales of a product or service.",13-1161.00
6,Data Scientists,"Develop and implement a set of techniques or analytics applications to transform raw data into meaningful information using data-oriented programming languages and visualization software.",15-2051.01
7,Sales Representatives,"Sell goods for wholesalers or manufacturers to businesses or groups of individuals.",41-4012.00
8,Accountants and Auditors,"Examine, analyze, and interpret accounting records to prepare financial statements, give advice, or audit and evaluate statements prepared by others.",13-2011.00
9,Human Resources Specialists,"Recruit, screen, interview, or place individuals within an organization.",13-1071.00
10,Web Developers,"Design, create, and modify websites. Analyze user needs to implement website content, graphics, performance, and capacity.",15-1134.00
EOF
print_success "Sample WORKBank data created"

# Create a minimal test script
print_step "Creating test script..."
cat > test_installation.py << 'EOF'
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
EOF

# Make test script executable
chmod +x test_installation.py

# Create a simple Makefile
print_step "Creating Makefile..."
cat > Makefile << 'EOF'
.PHONY: help test run clean activate

help:
	@echo "YC-WORKBank Pipeline Commands:"
	@echo "  make activate  - Activate virtual environment"
	@echo "  make test      - Test installation"
	@echo "  make run       - Run pipeline"
	@echo "  make clean     - Clean generated files"

activate:
	@echo "Run: source venv/bin/activate"

test:
	python test_installation.py

run:
	python cli.py run

clean:
	rm -rf output/* logs/*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
EOF

print_success "Makefile created"

# Test installation
print_step "Testing installation..."
$PYTHON_CMD test_installation.py

echo ""
echo "🎉 Installation Complete!"
echo "======================="
echo ""
echo "📁 Installation directory: $(pwd)"
echo ""
echo "🔧 Next Steps:"
echo "1. Add your API key to the .env file:"
echo "   nano .env"
echo ""
echo "2. Download the pipeline Python files and place them in this directory"
echo ""
echo "3. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "4. Test the pipeline:"
echo "   python cli.py --help"
echo ""
echo "5. Run the pipeline:"
echo "   python cli.py run"
echo ""
echo "6. Start the web interface:"
echo "   python web_interface.py"
echo ""
echo "📚 For detailed instructions, see the README.md file"
echo ""
print_warning "Don't forget to:"
echo "- Add your Anthropic or OpenAI API key to .env"
echo "- Replace sample data with actual WORKBank tasks"
echo "- Download all the Python pipeline files"

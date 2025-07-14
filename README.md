# YC-WORKBank Pipeline

An automated pipeline that maps Y Combinator (YC) startups to specific occupational tasks from the WORKBank database (derived from Department of Labor data).

## 🎯 Project Goals

This project builds an intelligent system to analyze the AI landscape in Y Combinator companies and map them to specific occupational tasks. The pipeline achieves the following objectives:

1. **Comprehensive Data Collection**: Collect the full list of YC companies (5,000+), including company names and detailed descriptions
2. **AI Classification**: Use Large Language Models (LLMs) to classify which companies are AI-related with binary classification
3. **Task Mapping**: For AI-related companies, map each company to relevant WORKBank occupational tasks using LLM-powered binary classification
4. **Structured Storage**: Store and maintain company-to-task mappings in structured formats (Google Sheets, CSV) for ongoing analysis
5. **Automation**: Automated pipeline that refreshes as new YC companies appear

## 📊 Data Sources

- **Y Combinator Company Data**: Public company directory with descriptions, batch information, and metadata
- **WORKBank Database**: Occupational task data derived from Department of Labor's O*NET database
- **Real-time Classification**: Anthropic Claude AI for intelligent company and task classification

## 🏗️ Architecture

```
YC Company Data → AI Classification → Task Mapping → Google Sheets/CSV Output
      ↓                ↓                ↓              ↓
   Web Scraping    Anthropic AI    WORKBank Tasks   Analytics
   Sample Data     Rule-based      Binary Class.    Reporting
```

## 🚀 Features

### Core Pipeline
- **Multi-source Data Collection**: Robust scraping with fallback mechanisms
- **AI-Powered Classification**: Uses Anthropic Claude for accurate AI company identification
- **Task Mapping**: Maps companies to 800+ occupational tasks from WORKBank
- **Google Sheets Integration**: Real-time data updates with multiple worksheets
- **CSV Export**: Timestamped data exports for analysis
- **Error Handling**: Graceful fallbacks and comprehensive logging

### Analytics & Reporting
- **Executive Summaries**: Automated analysis reports
- **Visual Charts**: Distribution analysis and trends (coming soon)
- **Performance Metrics**: Classification accuracy and coverage statistics
- **Historical Tracking**: Monitor changes over time

### Automation
- **Scheduled Runs**: Daily/weekly automated updates
- **Incremental Processing**: Only process new companies
- **Rate Limiting**: Respectful API usage
- **Monitoring**: Health checks and alerting

## 📁 Project Structure

```
yc-workbank-pipeline/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.template            # Environment variables template
├── .gitignore              # Git ignore rules
│
├── Core Pipeline Files/
│   ├── yc_fixed_scraper.py     # Enhanced YC company scraper
│   ├── yc_pipeline_enhanced.py # Main pipeline with all features
│   ├── yc_scraper.py           # Original Google Sheets integration
│   ├── cli.py                  # Command-line interface
│   └── config.py               # Configuration management
│
├── Testing & Setup/
│   ├── quick_test.py           # Pipeline testing script
│   ├── run_examples.py         # Example usage scenarios
│   └── install.sh              # Installation script
│
├── Data/
│   └── workbank_tasks.csv      # Sample WORKBank occupational tasks
│
└── Output/
    ├── analytics/              # Generated charts and reports
    ├── *.csv                   # Company data exports
    └── logs/                   # Pipeline execution logs
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git
- Google Cloud Platform account (for Sheets API)
- Anthropic API key

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/george550/yc-workbank-pipeline.git
   cd yc-workbank-pipeline
   ```

2. **Set up virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and configuration
   ```

5. **Set up Google Sheets API**
   - Create a Google Cloud Project
   - Enable Google Sheets API
   - Create a service account and download `credentials.json`
   - Place `credentials.json` in the project root

6. **Test the installation**
   ```bash
   python yc_pipeline_enhanced.py --companies 10
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_SHEET_ID=your_google_sheet_id_here

# Pipeline Settings
MAX_COMPANIES=50
ENABLE_REAL_SCRAPING=true
ENABLE_TASK_MAPPING=true
ENABLE_ANALYTICS=true

# Rate Limiting
API_RATE_LIMIT_DELAY=1.0
SCRAPING_DELAY=0.5
BATCH_SIZE=10
```

### Google Sheets Setup

1. Create a new Google Sheet
2. Share it with your service account email (from `credentials.json`)
3. Copy the Sheet ID from the URL
4. Add the ID to your `.env` file

## 📊 Usage

### Basic Pipeline Run
```bash
# Test with 10 companies
python yc_pipeline_enhanced.py --companies 10

# Full run with 50 companies
python yc_pipeline_enhanced.py --companies 50
```

### Command Line Interface
```bash
# Interactive examples menu
python run_examples.py

# Quick pipeline test
python cli.py run

# Setup verification
python quick_test.py
```

### Pipeline Modes
- **Test Mode**: Process 10 companies for quick validation
- **Full Mode**: Complete analysis with specified company count
- **Incremental**: Only process new companies since last run
- **Analytics Only**: Generate reports from existing data

## 📈 Output

### Google Sheets
The pipeline creates multiple worksheets:
- **YC Companies**: Complete company data with AI classification
- **Summary**: Statistics and data source breakdown
- **Task Mappings**: Company-to-occupation task relationships (if enabled)

### CSV Files
- `output/enhanced_companies_YYYYMMDD_HHMMSS.csv`: Complete company dataset
- `output/task_mappings_YYYYMMDD_HHMMSS.csv`: Task mapping results
- `output/analysis_report_YYYYMMDD_HHMMSS.txt`: Executive summary

### Sample Results
```
🎯 Enhanced Pipeline Results:
Total companies: 50
AI-related: 15 (30.0%)
CSV saved: output/enhanced_companies_20250715_120000.csv
Google Sheets: ✅ Updated

AI Companies:
  - OpenAI
  - Anthropic
  - Scale AI
  - Hugging Face
  - Weights & Biases
```

## 🤖 AI Classification

The pipeline uses sophisticated prompts to classify companies:

### AI Company Classification
```
Analyze this company and determine if it is AI-related.

Company: [Name]
Description: [Description]

Consider a company AI-related if it:
- Develops AI/ML models or algorithms
- Provides AI-powered products or services
- Focuses on machine learning, NLP, computer vision, or robotics
- Uses AI as a core part of their business model

Respond with only 'true' or 'false'.
```

### Task Mapping (Future Enhancement)
For AI companies, the system will map them to specific WORKBank occupational tasks using binary classification for each of 800+ tasks.

## 🔄 Automation

### Scheduled Runs
```bash
# Start automated scheduling
python enhanced_yc_pipeline.py --mode schedule

# Configure in .env:
FULL_PIPELINE_SCHEDULE=02:00      # Daily at 2 AM
INCREMENTAL_UPDATE_SCHEDULE=06:00  # Daily at 6 AM
```

### Monitoring
- Health checks for API connectivity
- Data freshness validation
- Error tracking and alerting
- Performance metrics

## 🧪 Testing

### Quick Tests
```bash
# Test all components
python quick_test.py

# Test scraping only
python yc_fixed_scraper.py

# Test Google Sheets integration
python yc_scraper.py
```

### Validation
- API connectivity tests
- Data quality checks
- Classification accuracy validation
- Output format verification

## 📊 Research Background

This pipeline implements the methodology described in academic research on AI's impact on occupational tasks. The system replicates and extends published approaches for:

- Large-scale company classification
- Occupational task mapping
- AI impact analysis on labor markets
- Longitudinal tracking of AI adoption trends

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **Repository**: https://github.com/george550/yc-workbank-pipeline
- **Issues**: https://github.com/george550/yc-workbank-pipeline/issues
- **Anthropic API**: https://console.anthropic.com
- **Google Sheets API**: https://developers.google.com/sheets/api

## 📞 Support

For questions or support:
- Open an issue on GitHub
- Check the examples in `run_examples.py`
- Review the test scripts for troubleshooting

---

**Status**: ✅ Active Development | **Last Updated**: July 2025 | **Version**: 1.0

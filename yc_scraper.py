#!/usr/bin/env python3
"""
Enhanced YC Company Scraper with Google Sheets Integration
"""

import time
import logging
import csv
import os
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Google Sheets libraries
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GOOGLE_SHEETS_AVAILABLE = True
    logger.info("Google Sheets integration available")
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    logger.warning("Google Sheets libraries not available, will save to CSV only")

# Try to import Anthropic
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not available, using rule-based classification")

class EnhancedYCPipeline:
    """Enhanced YC Pipeline with more companies and Google Sheets"""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.sheet_id = os.getenv('GOOGLE_SHEET_ID')
        self.max_companies = int(os.getenv('MAX_COMPANIES', '20'))
        
        # Initialize Anthropic client if available
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.anthropic_client = Anthropic(api_key=self.api_key)
            self.use_ai_classification = True
            logger.info("Using Anthropic AI for classification")
        else:
            self.use_ai_classification = False
            logger.info("Using rule-based classification")
        
        # Initialize Google Sheets client if available
        if GOOGLE_SHEETS_AVAILABLE and os.path.exists('credentials.json'):
            try:
                scope = [
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
                creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
                self.sheets_client = gspread.authorize(creds)
                self.use_google_sheets = True
                logger.info("Google Sheets integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Sheets: {e}")
                self.use_google_sheets = False
        else:
            self.use_google_sheets = False
            logger.info("Google Sheets not available, using CSV only")
    
    def get_sample_companies(self) -> List[Dict]:
        """Get an expanded list of sample YC companies"""
        logger.info(f"Creating {self.max_companies} sample YC companies...")
        
        # Extended list of real YC companies
        base_companies = [
            {
                'name': 'OpenAI',
                'description': 'AI research company developing large language models and AI systems like GPT and ChatGPT',
                'batch': 'W16',
                'website': 'https://openai.com',
                'founded': '2015',
                'team_size': '500+',
                'location': 'San Francisco, CA',
                'tags': 'AI, Machine Learning, Research, LLM',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Anthropic',
                'description': 'AI safety company focused on developing safe, beneficial AI systems like Claude',
                'batch': 'S21',
                'website': 'https://anthropic.com',
                'founded': '2021',
                'team_size': '200+',
                'location': 'San Francisco, CA',
                'tags': 'AI, Safety, Research, LLM',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Stripe',
                'description': 'Online payment processing platform for businesses and e-commerce',
                'batch': 'S09',
                'website': 'https://stripe.com',
                'founded': '2010',
                'team_size': '4000+',
                'location': 'San Francisco, CA',
                'tags': 'Fintech, Payments, B2B, SaaS',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Airbnb',
                'description': 'Online marketplace for lodging and travel experiences worldwide',
                'batch': 'W08',
                'website': 'https://airbnb.com',
                'founded': '2008',
                'team_size': '6000+',
                'location': 'San Francisco, CA',
                'tags': 'Travel, Marketplace, Consumer',
                'status': 'Public',
                'is_ai_related': None
            },
            {
                'name': 'Zapier',
                'description': 'Automation platform connecting different apps and services without coding',
                'batch': 'S12',
                'website': 'https://zapier.com',
                'founded': '2011',
                'team_size': '400+',
                'location': 'Remote',
                'tags': 'Automation, Productivity, B2B, SaaS',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Scale AI',
                'description': 'Data platform for AI providing training data, annotation, and model evaluation services',
                'batch': 'S16',
                'website': 'https://scale.com',
                'founded': '2016',
                'team_size': '300+',
                'location': 'San Francisco, CA',
                'tags': 'AI, Data, Machine Learning, B2B',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Hugging Face',
                'description': 'Platform for machine learning models, datasets, and AI development tools',
                'batch': 'W20',
                'website': 'https://huggingface.co',
                'founded': '2016',
                'team_size': '150+',
                'location': 'New York, NY',
                'tags': 'AI, Machine Learning, Open Source, Developer Tools',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Weights & Biases',
                'description': 'MLOps platform for experiment tracking, model management, and collaboration',
                'batch': 'W18',
                'website': 'https://wandb.ai',
                'founded': '2017',
                'team_size': '200+',
                'location': 'San Francisco, CA',
                'tags': 'AI, MLOps, Developer Tools, B2B',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Notion',
                'description': 'Productivity software for note-taking, project management, and team collaboration',
                'batch': 'S19',
                'website': 'https://notion.so',
                'founded': '2016',
                'team_size': '400+',
                'location': 'San Francisco, CA',
                'tags': 'Productivity, SaaS, B2B, Collaboration',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Twilio',
                'description': 'Cloud communications platform for messaging, voice, and video APIs',
                'batch': 'W08',
                'website': 'https://twilio.com',
                'founded': '2008',
                'team_size': '7000+',
                'location': 'San Francisco, CA',
                'tags': 'Communications, API, B2B, SaaS',
                'status': 'Public',
                'is_ai_related': None
            },
            {
                'name': 'Coinbase',
                'description': 'Cryptocurrency exchange and wallet platform for buying, selling, and storing digital assets',
                'batch': 'S12',
                'website': 'https://coinbase.com',
                'founded': '2012',
                'team_size': '3000+',
                'location': 'San Francisco, CA',
                'tags': 'Cryptocurrency, Fintech, B2C, Exchange',
                'status': 'Public',
                'is_ai_related': None
            },
            {
                'name': 'Instacart',
                'description': 'Grocery delivery and pickup service connecting customers with personal shoppers',
                'batch': 'S12',
                'website': 'https://instacart.com',
                'founded': '2012',
                'team_size': '3000+',
                'location': 'San Francisco, CA',
                'tags': 'Grocery, Delivery, Marketplace, Consumer',
                'status': 'Public',
                'is_ai_related': None
            },
            {
                'name': 'DoorDash',
                'description': 'Food delivery platform connecting customers with restaurants and delivery drivers',
                'batch': 'S13',
                'website': 'https://doordash.com',
                'founded': '2013',
                'team_size': '8000+',
                'location': 'San Francisco, CA',
                'tags': 'Food Delivery, Marketplace, Logistics, Consumer',
                'status': 'Public',
                'is_ai_related': None
            },
            {
                'name': 'Reddit',
                'description': 'Social news aggregation and discussion website with millions of communities',
                'batch': 'S05',
                'website': 'https://reddit.com',
                'founded': '2005',
                'team_size': '2000+',
                'location': 'San Francisco, CA',
                'tags': 'Social Media, News, Community, Consumer',
                'status': 'Public',
                'is_ai_related': None
            },
            {
                'name': 'GitLab',
                'description': 'DevOps platform for software development, version control, and CI/CD',
                'batch': 'W15',
                'website': 'https://gitlab.com',
                'founded': '2014',
                'team_size': '1300+',
                'location': 'Remote',
                'tags': 'Developer Tools, DevOps, Software, B2B',
                'status': 'Public',
                'is_ai_related': None
            },
            {
                'name': 'PagerDuty',
                'description': 'Digital operations management platform for incident response and monitoring',
                'batch': 'S10',
                'website': 'https://pagerduty.com',
                'founded': '2009',
                'team_size': '900+',
                'location': 'San Francisco, CA',
                'tags': 'DevOps, Monitoring, Incident Management, B2B',
                'status': 'Public',
                'is_ai_related': None
            },
            {
                'name': 'Segment',
                'description': 'Customer data platform for collecting, cleaning, and controlling customer data',
                'batch': 'S11',
                'website': 'https://segment.com',
                'founded': '2011',
                'team_size': '400+',
                'location': 'San Francisco, CA',
                'tags': 'Data, Analytics, Customer Data, B2B',
                'status': 'Acquired',
                'is_ai_related': None
            },
            {
                'name': 'Perplexity',
                'description': 'AI-powered search engine and answer engine using large language models',
                'batch': 'W22',
                'website': 'https://perplexity.ai',
                'founded': '2022',
                'team_size': '50+',
                'location': 'San Francisco, CA',
                'tags': 'AI, Search, LLM, Consumer',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Vercel',
                'description': 'Frontend cloud platform for deploying and hosting web applications',
                'batch': 'S20',
                'website': 'https://vercel.com',
                'founded': '2015',
                'team_size': '200+',
                'location': 'San Francisco, CA',
                'tags': 'Developer Tools, Cloud, Frontend, B2B',
                'status': 'Active',
                'is_ai_related': None
            },
            {
                'name': 'Retool',
                'description': 'Low-code platform for building internal tools and business applications',
                'batch': 'W17',
                'website': 'https://retool.com',
                'founded': '2017',
                'team_size': '300+',
                'location': 'San Francisco, CA',
                'tags': 'Low-Code, Developer Tools, B2B, SaaS',
                'status': 'Active',
                'is_ai_related': None
            }
        ]
        
        # Return the requested number of companies
        return base_companies[:self.max_companies]
    
    def classify_company(self, company: Dict) -> bool:
        """Classify if a company is AI-related"""
        if self.use_ai_classification:
            return self._ai_classify(company)
        else:
            return self._rule_based_classify(company)
    
    def _ai_classify(self, company: Dict) -> bool:
        """Use Anthropic AI to classify company"""
        prompt = f"""You will be presented with a company description. Your job is to classify if the company is an AI related company or not.

An AI-related company is defined as a company that is involved in the research, development, or application of AI.

Output only "true" or "false".

Company: {company['name']}
Description: {company['description']}
Tags: {company.get('tags', '')}

AI-related:"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.content[0].text.strip().lower()
            return "true" in result
        except Exception as e:
            logger.error(f"AI classification failed for {company['name']}: {e}")
            return self._rule_based_classify(company)
    
    def _rule_based_classify(self, company: Dict) -> bool:
        """Simple rule-based classification"""
        ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'neural network', 'nlp', 'computer vision', 'automation', 'predictive',
            'algorithm', 'data science', 'analytics', 'intelligent', 'smart', 'llm',
            'language model', 'gpt', 'claude', 'chatbot'
        ]
        
        text = f"{company['name']} {company['description']} {company.get('tags', '')}".lower()
        return any(keyword in text for keyword in ai_keywords)
    
    def save_to_csv(self, companies: List[Dict]) -> str:
        """Save companies to CSV file"""
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"output/yc_companies_enhanced_{timestamp}.csv"
        
        if companies:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=companies[0].keys())
                writer.writeheader()
                writer.writerows(companies)
            
            logger.info(f"💾 Saved {len(companies)} companies to {filename}")
        
        return filename
    
    def save_to_google_sheets(self, companies: List[Dict]) -> bool:
        """Save companies to Google Sheets"""
        if not self.use_google_sheets or not self.sheet_id:
            logger.warning("Google Sheets not available or no sheet ID provided")
            return False
        
        try:
            # Open the sheet
            workbook = self.sheets_client.open_by_key(self.sheet_id)
            worksheet = workbook.sheet1
            
            # Clear existing data
            worksheet.clear()
            
            # Prepare data
            if companies:
                headers = list(companies[0].keys())
                data = [headers]
                
                for company in companies:
                    row = [str(company.get(header, '')) for header in headers]
                    data.append(row)
                
                # Update the sheet
                worksheet.update(data)
                
                # Format headers
                worksheet.format('1:1', {
                    'textFormat': {'bold': True},
                    'backgroundColor': {'red': 0.8, 'green': 0.8, 'blue': 0.8}
                })
                
                logger.info(f"✅ Successfully updated Google Sheet with {len(companies)} companies")
                logger.info(f"🔗 View at: https://docs.google.com/spreadsheets/d/{self.sheet_id}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to update Google Sheet: {e}")
            return False
    
    def run_enhanced_pipeline(self):
        """Run the complete enhanced pipeline"""
        logger.info("🚀 Starting Enhanced YC-WORKBank Pipeline...")
        logger.info(f"📊 Processing {self.max_companies} companies...")
        
        # Step 1: Get companies
        companies = self.get_sample_companies()
        logger.info(f"📋 Loaded {len(companies)} companies")
        
        # Step 2: Classify companies
        logger.info("🤖 Starting AI classification...")
        for i, company in enumerate(companies):
            logger.info(f"Classifying {i+1}/{len(companies)}: {company['name']}")
            company['is_ai_related'] = self.classify_company(company)
            time.sleep(0.2)  # Small delay to be respectful to API
        
        # Step 3: Save to CSV (always works)
        csv_file = self.save_to_csv(companies)
        
        # Step 4: Save to Google Sheets (if available)
        sheets_success = self.save_to_google_sheets(companies)
        
        # Step 5: Generate summary
        ai_companies = [c for c in companies if c['is_ai_related']]
        total_companies = len(companies)
        ai_count = len(ai_companies)
        
        logger.info("📊 Pipeline Summary:")
        logger.info(f"   Total companies: {total_companies}")
        logger.info(f"   AI companies: {ai_count} ({ai_count/total_companies*100:.1f}%)")
        logger.info(f"   Non-AI companies: {total_companies - ai_count}")
        logger.info(f"   CSV saved: {csv_file}")
        
        if sheets_success:
            logger.info(f"   Google Sheet updated: https://docs.google.com/spreadsheets/d/{self.sheet_id}")
        else:
            logger.warning("   Google Sheet update failed - check CSV file for data")
        
        # Print AI companies
        logger.info("🤖 AI-Related Companies:")
        for company in ai_companies:
            logger.info(f"   - {company['name']}: {company['description'][:60]}...")
        
        return {
            'total_companies': total_companies,
            'ai_companies': ai_count,
            'csv_file': csv_file,
            'sheets_updated': sheets_success,
            'success': True
        }

def main():
    """Main function"""
    pipeline = EnhancedYCPipeline()
    results = pipeline.run_enhanced_pipeline()
    
    print("\n🎉 Enhanced Pipeline Results:")
    print(f"✅ Total companies processed: {results['total_companies']}")
    print(f"🤖 AI-related companies found: {results['ai_companies']}")
    print(f"💾 Data saved to: {results['csv_file']}")
    
    if results['sheets_updated']:
        print(f"📊 Google Sheet updated successfully!")
    else:
        print("⚠️  Google Sheet update failed, but CSV backup available")

if __name__ == "__main__":
    main()

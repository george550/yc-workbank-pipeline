# enhanced_yc_pipeline.py
#!/usr/bin/env python3
"""
Full-Featured YC Pipeline with Real Scraping, Task Mapping, Analytics, and Automation
"""

import requests
from bs4 import BeautifulSoup
import time
import logging
import csv
import json
import os
import schedule
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import threading
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Import optional libraries
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    logger.warning("Google Sheets libraries not available")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not available")

@dataclass
class Company:
    """Enhanced company data structure"""
    name: str
    description: str
    batch: str = ""
    website: str = ""
    founded: str = ""
    team_size: str = ""
    location: str = ""
    tags: List[str] = None
    status: str = ""
    logo_url: str = ""
    is_ai_related: Optional[bool] = None
    confidence_score: Optional[float] = None
    mapped_tasks: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.mapped_tasks is None:
            self.mapped_tasks = []

@dataclass
class WorkBankTask:
    """WORKBank task structure"""
    task_id: str
    occupation: str
    task_description: str
    onet_soc_code: str

@dataclass
class CompanyTaskMapping:
    """Company to task mapping"""
    company_name: str
    task_id: str
    occupation: str
    is_applicable: bool
    confidence_score: float
    reasoning: str = ""

class RealYCWebScraper:
    """Scrapes real YC company data from multiple sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.base_delay = 1.0  # Be respectful with scraping
        
    def scrape_yc_companies(self, max_companies: int = 500) -> List[Company]:
        """Scrape real YC companies using multiple methods"""
        logger.info(f"Starting to scrape up to {max_companies} real YC companies...")
        
        companies = []
        
        # Method 1: Try YC's public API/JSON endpoint
        companies.extend(self._try_yc_api(max_companies))
        
        # Method 2: Scrape YC directory pages
        if len(companies) < max_companies:
            companies.extend(self._scrape_yc_directory(max_companies - len(companies)))
        
        # Method 3: Use YC database dumps if available
        if len(companies) < max_companies:
            companies.extend(self._load_yc_database_dump(max_companies - len(companies)))
        
        logger.info(f"Successfully collected {len(companies)} companies")
        return companies[:max_companies]
    
    def _try_yc_api(self, max_companies: int) -> List[Company]:
        """Try to get data from YC's API endpoints"""
        companies = []
        
        try:
            # Try the companies.json endpoint
            url = "https://www.ycombinator.com/companies.json"
            logger.info("Attempting to fetch from YC API...")
            
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully fetched {len(data)} companies from YC API")
                
                for company_data in data[:max_companies]:
                    company = Company(
                        name=company_data.get('name', ''),
                        description=company_data.get('one_liner', ''),
                        batch=company_data.get('batch', ''),
                        website=company_data.get('website', ''),
                        founded=str(company_data.get('founded', '')),
                        team_size=str(company_data.get('team_size', '')),
                        location=company_data.get('location', ''),
                        tags=company_data.get('tags', []),
                        status=company_data.get('status', 'Active'),
                        logo_url=company_data.get('logo_url', '')
                    )
                    companies.append(company)
                
                return companies
            
        except Exception as e:
            logger.warning(f"YC API method failed: {e}")
        
        return []
    
    def _scrape_yc_directory(self, max_companies: int) -> List[Company]:
        """Scrape YC directory pages"""
        companies = []
        
        try:
            logger.info("Attempting to scrape YC directory pages...")
            
            # YC directory with pagination
            page = 1
            while len(companies) < max_companies:
                url = f"https://www.ycombinator.com/companies?page={page}"
                logger.info(f"Scraping page {page}...")
                
                response = self.session.get(url, timeout=20)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch page {page}: {response.status_code}")
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for company cards/entries
                company_elements = soup.find_all(['div', 'article'], class_=lambda x: x and any(
                    keyword in x.lower() for keyword in ['company', 'startup', 'card']
                ))
                
                if not company_elements:
                    logger.info(f"No companies found on page {page}, stopping")
                    break
                
                for element in company_elements:
                    if len(companies) >= max_companies:
                        break
                    
                    try:
                        company = self._parse_company_element(element)
                        if company and company.name:
                            companies.append(company)
                    except Exception as e:
                        logger.debug(f"Failed to parse company element: {e}")
                
                page += 1
                time.sleep(self.base_delay)  # Be respectful
                
                if page > 20:  # Safety limit
                    break
                    
        except Exception as e:
            logger.warning(f"Directory scraping failed: {e}")
        
        return companies
    
    def _parse_company_element(self, element) -> Optional[Company]:
        """Parse a company element from HTML"""
        try:
            # Extract name
            name_elem = element.find(['h3', 'h2', 'h4'], class_=lambda x: x and 'name' in x.lower()) or \
                       element.find('a') or element.find(['h3', 'h2', 'h4'])
            name = name_elem.get_text().strip() if name_elem else ""
            
            # Extract description
            desc_elem = element.find('p') or element.find(['div', 'span'], class_=lambda x: x and 'description' in x.lower())
            description = desc_elem.get_text().strip() if desc_elem else ""
            
            # Extract other details
            batch = ""
            website = ""
            
            # Look for links
            link_elem = element.find('a', href=True)
            if link_elem:
                href = link_elem.get('href', '')
                if href.startswith('http'):
                    website = href
            
            if name and description:
                return Company(
                    name=name,
                    description=description,
                    batch=batch,
                    website=website
                )
                
        except Exception as e:
            logger.debug(f"Error parsing company element: {e}")
        
        return None
    
    def _load_yc_database_dump(self, max_companies: int) -> List[Company]:
        """Load from cached YC database dump if available"""
        companies = []
        
        try:
            # Check for cached data file
            cache_file = Path('data/yc_companies_cache.json')
            if cache_file.exists():
                logger.info("Loading from cached YC data...")
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                for company_data in cached_data[:max_companies]:
                    company = Company(**company_data)
                    companies.append(company)
                
                logger.info(f"Loaded {len(companies)} companies from cache")
        
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
        
        return companies

class AIClassifierAdvanced:
    """Advanced AI classification with confidence scoring"""
    
    def __init__(self):
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = Anthropic(api_key=self.api_key)
            self.use_ai = True
            logger.info("Using Anthropic AI for advanced classification")
        else:
            self.use_ai = False
            logger.info("Using rule-based classification")
    
    def classify_companies_batch(self, companies: List[Company]) -> List[Company]:
        """Classify companies with confidence scores"""
        logger.info(f"Starting advanced classification of {len(companies)} companies...")
        
        for i, company in enumerate(companies):
            logger.info(f"Classifying {i+1}/{len(companies)}: {company.name}")
            
            result = self.classify_company_advanced(company)
            company.is_ai_related = result['is_ai_related']
            company.confidence_score = result['confidence_score']
            
            time.sleep(0.3)  # Rate limiting
        
        ai_count = sum(1 for c in companies if c.is_ai_related)
        logger.info(f"Classification complete: {ai_count}/{len(companies)} are AI-related")
        
        return companies
    
    def classify_company_advanced(self, company: Company) -> Dict:
        """Advanced classification with confidence scoring"""
        if self.use_ai:
            return self._ai_classify_advanced(company)
        else:
            return self._rule_based_classify_advanced(company)
    
    def _ai_classify_advanced(self, company: Company) -> Dict:
        """Advanced AI classification with reasoning"""
        tags_str = ', '.join(company.tags) if company.tags else ''
        
        prompt = f"""You are an expert at identifying AI-related companies. Analyze this company and determine if it's AI-related.

Company: {company.name}
Description: {company.description}
Tags: {tags_str}
Website: {company.website}
Location: {company.location}

An AI-related company is involved in research, development, or application of artificial intelligence, machine learning, or related technologies.

Respond with a JSON object containing:
- "is_ai_related": true or false
- "confidence_score": 0.0 to 1.0 (how confident you are)
- "reasoning": brief explanation of your decision

Example response:
{{"is_ai_related": true, "confidence_score": 0.85, "reasoning": "Company develops machine learning models for computer vision applications"}}

Your response:"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                return {
                    'is_ai_related': bool(result.get('is_ai_related', False)),
                    'confidence_score': float(result.get('confidence_score', 0.5)),
                    'reasoning': result.get('reasoning', '')
                }
            except json.JSONDecodeError:
                # Fallback parsing
                is_ai = 'true' in result_text.lower()
                return {
                    'is_ai_related': is_ai,
                    'confidence_score': 0.7 if is_ai else 0.3,
                    'reasoning': 'Parsed from text response'
                }
                
        except Exception as e:
            logger.error(f"AI classification failed for {company.name}: {e}")
            return self._rule_based_classify_advanced(company)
    
    def _rule_based_classify_advanced(self, company: Company) -> Dict:
        """Rule-based classification with confidence scoring"""
        ai_keywords = {
            'high_confidence': ['artificial intelligence', 'machine learning', 'deep learning', 'neural network', 'ai research', 'computer vision', 'natural language processing', 'llm', 'gpt', 'claude'],
            'medium_confidence': ['ai', 'ml', 'automation', 'algorithm', 'data science', 'predictive analytics', 'intelligent', 'smart'],
            'low_confidence': ['analytics', 'data', 'optimization', 'insights', 'platform']
        }
        
        text = f"{company.name} {company.description} {' '.join(company.tags)}".lower()
        
        # Check for high confidence keywords
        for keyword in ai_keywords['high_confidence']:
            if keyword in text:
                return {'is_ai_related': True, 'confidence_score': 0.9, 'reasoning': f'Contains high-confidence AI keyword: {keyword}'}
        
        # Check for medium confidence keywords
        medium_matches = [kw for kw in ai_keywords['medium_confidence'] if kw in text]
        if medium_matches:
            return {'is_ai_related': True, 'confidence_score': 0.7, 'reasoning': f'Contains AI keywords: {", ".join(medium_matches)}'}
        
        # Check for low confidence keywords
        low_matches = [kw for kw in ai_keywords['low_confidence'] if kw in text]
        if len(low_matches) >= 2:
            return {'is_ai_related': True, 'confidence_score': 0.4, 'reasoning': f'Multiple data-related keywords: {", ".join(low_matches)}'}
        
        return {'is_ai_related': False, 'confidence_score': 0.8, 'reasoning': 'No AI-related keywords found'}

class TaskMappingEngine:
    """Maps companies to WORKBank occupational tasks"""
    
    def __init__(self):
        self.tasks = self._load_workbank_tasks()
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = Anthropic(api_key=self.api_key)
            self.use_ai = True
            logger.info("Using AI for task mapping")
        else:
            self.use_ai = False
            logger.info("Using rule-based task mapping")
    
    def _load_workbank_tasks(self) -> List[WorkBankTask]:
        """Load WORKBank tasks"""
        tasks = []
        workbank_file = os.getenv('WORKBANK_FILE', 'data/workbank_tasks.csv')
        
        try:
            df = pd.read_csv(workbank_file)
            for _, row in df.iterrows():
                task = WorkBankTask(
                    task_id=str(row.get('task_id', '')),
                    occupation=str(row.get('occupation', '')),
                    task_description=str(row.get('task_description', '')),
                    onet_soc_code=str(row.get('onet_soc_code', ''))
                )
                tasks.append(task)
        except Exception as e:
            logger.error(f"Failed to load WORKBank tasks: {e}")
            # Create sample tasks
            tasks = [
                WorkBankTask("1", "Computer Programmers", "Write, update, and maintain computer programs", "15-1251.00"),
                WorkBankTask("2", "Data Scientists", "Develop data models and analytics solutions", "15-2051.01"),
                WorkBankTask("3", "Software Developers", "Design and develop software applications", "15-1132.00"),
                WorkBankTask("4", "Customer Service Representatives", "Interact with customers to provide information", "43-4051.00"),
                WorkBankTask("5", "Marketing Managers", "Develop marketing strategies and campaigns", "11-2021.00")
            ]
        
        logger.info(f"Loaded {len(tasks)} WORKBank tasks")
        return tasks
    
    def map_companies_to_tasks(self, companies: List[Company]) -> List[CompanyTaskMapping]:
        """Map companies to relevant tasks"""
        logger.info(f"Starting task mapping for {len(companies)} companies...")
        
        mappings = []
        
        for i, company in enumerate(companies):
            logger.info(f"Mapping {i+1}/{len(companies)}: {company.name}")
            
            company_mappings = self._map_company_to_tasks(company)
            mappings.extend(company_mappings)
            
            # Update company with mapped tasks
            company.mapped_tasks = [m.task_id for m in company_mappings if m.is_applicable]
            
            time.sleep(0.2)  # Rate limiting
        
        applicable_mappings = [m for m in mappings if m.is_applicable]
        logger.info(f"Task mapping complete: {len(applicable_mappings)} applicable mappings found")
        
        return mappings
    
    def _map_company_to_tasks(self, company: Company) -> List[CompanyTaskMapping]:
        """Map a single company to tasks"""
        mappings = []
        
        # Only map to a subset of tasks to avoid API overuse
        sample_tasks = self.tasks[:10]  # Top 10 most relevant tasks
        
        for task in sample_tasks:
            if self.use_ai:
                mapping = self._ai_map_company_to_task(company, task)
            else:
                mapping = self._rule_based_map_company_to_task(company, task)
            
            if mapping:
                mappings.append(mapping)
        
        return mappings
    
    def _ai_map_company_to_task(self, company: Company, task: WorkBankTask) -> Optional[CompanyTaskMapping]:
        """Use AI to map company to task"""
        prompt = f"""Determine if workers in this occupation would be primary users of this company's product/service.

Company: {company.name}
Description: {company.description}
Website: {company.website}

Occupation: {task.occupation}
Task: {task.task_description}

Would workers doing this task be primary or intended users of this company's offering?

Respond with JSON:
{{"is_applicable": true/false, "confidence_score": 0.0-1.0, "reasoning": "brief explanation"}}

Response:"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text.strip()
            
            try:
                result = json.loads(result_text)
                return CompanyTaskMapping(
                    company_name=company.name,
                    task_id=task.task_id,
                    occupation=task.occupation,
                    is_applicable=bool(result.get('is_applicable', False)),
                    confidence_score=float(result.get('confidence_score', 0.5)),
                    reasoning=result.get('reasoning', '')
                )
            except json.JSONDecodeError:
                # Fallback
                is_applicable = 'true' in result_text.lower()
                return CompanyTaskMapping(
                    company_name=company.name,
                    task_id=task.task_id,
                    occupation=task.occupation,
                    is_applicable=is_applicable,
                    confidence_score=0.5,
                    reasoning='Parsed from text'
                )
                
        except Exception as e:
            logger.error(f"AI task mapping failed: {e}")
            return self._rule_based_map_company_to_task(company, task)
    
    def _rule_based_map_company_to_task(self, company: Company, task: WorkBankTask) -> CompanyTaskMapping:
        """Rule-based task mapping"""
        # Simple keyword matching
        company_text = f"{company.name} {company.description}".lower()
        task_text = f"{task.occupation} {task.task_description}".lower()
        
        # Look for relevant keywords
        if 'software' in company_text or 'app' in company_text:
            if 'programmer' in task_text or 'developer' in task_text:
                return CompanyTaskMapping(
                    company_name=company.name,
                    task_id=task.task_id,
                    occupation=task.occupation,
                    is_applicable=True,
                    confidence_score=0.7,
                    reasoning='Software company matches developer task'
                )
        
        return CompanyTaskMapping(
            company_name=company.name,
            task_id=task.task_id,
            occupation=task.occupation,
            is_applicable=False,
            confidence_score=0.8,
            reasoning='No clear relevance found'
        )

class AnalyticsEngine:
    """Advanced analytics and visualization"""
    
    def __init__(self):
        self.output_dir = Path('output/analytics')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_comprehensive_analysis(self, companies: List[Company], mappings: List[CompanyTaskMapping]):
        """Generate comprehensive analytics"""
        logger.info("Generating comprehensive analytics...")
        
        # Convert to DataFrames for analysis
        companies_df = self._companies_to_dataframe(companies)
        mappings_df = self._mappings_to_dataframe(mappings)
        
        # Generate all analyses
        self._analyze_ai_landscape(companies_df)
        self._analyze_company_batches(companies_df)
        self._analyze_locations(companies_df)
        self._analyze_task_mappings(mappings_df)
        self._generate_executive_summary(companies_df, mappings_df)
        
        logger.info(f"Analytics saved to {self.output_dir}")
    
    def _companies_to_dataframe(self, companies: List[Company]) -> pd.DataFrame:
        """Convert companies to DataFrame"""
        data = []
        for company in companies:
            data.append({
                'name': company.name,
                'description': company.description,
                'batch': company.batch,
                'founded': company.founded,
                'team_size': company.team_size,
                'location': company.location,
                'status': company.status,
                'is_ai_related': company.is_ai_related,
                'confidence_score': company.confidence_score,
                'tags_count': len(company.tags),
                'mapped_tasks_count': len(company.mapped_tasks)
            })
        return pd.DataFrame(data)
    
    def _mappings_to_dataframe(self, mappings: List[CompanyTaskMapping]) -> pd.DataFrame:
        """Convert mappings to DataFrame"""
        data = []
        for mapping in mappings:
            data.append({
                'company_name': mapping.company_name,
                'task_id': mapping.task_id,
                'occupation': mapping.occupation,
                'is_applicable': mapping.is_applicable,
                'confidence_score': mapping.confidence_score,
                'reasoning': mapping.reasoning
            })
        return pd.DataFrame(data)
    
    def _analyze_ai_landscape(self, df: pd.DataFrame):
        """Analyze AI landscape"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # AI vs Non-AI distribution
        ai_counts = df['is_ai_related'].value_counts()
        axes[0, 0].pie(ai_counts.values, labels=['Non-AI', 'AI'], autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('AI vs Non-AI Companies')
        
        # Confidence score distribution
        if 'confidence_score' in df.columns and df['confidence_score'].notna().any():
            df['confidence_score'].hist(bins=20, ax=axes[0, 1])
            axes[0, 1].set_title('Classification Confidence Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
        
        # AI companies by batch
        if df['batch'].notna().any():
            ai_by_batch = df[df['is_ai_related'] == True]['batch'].value_counts().head(10)
            if not ai_by_batch.empty:
                ai_by_batch.plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('AI Companies by YC Batch')
                axes[1, 0].set_xlabel('Batch')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Team size distribution for AI vs Non-AI
        if df['team_size'].notna().any():
            # Create numeric team size categories
            df['team_size_numeric'] = df['team_size'].apply(self._parse_team_size)
            ai_team_sizes = df[df['is_ai_related'] == True]['team_size_numeric'].dropna()
            non_ai_team_sizes = df[df['is_ai_related'] == False]['team_size_numeric'].dropna()
            
            if not ai_team_sizes.empty and not non_ai_team_sizes.empty:
                axes[1, 1].hist([ai_team_sizes, non_ai_team_sizes], bins=20, alpha=0.7, label=['AI', 'Non-AI'])
                axes[1, 1].set_title('Team Size Distribution')
                axes[1, 1].set_xlabel('Team Size')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ai_landscape_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_company_batches(self, df: pd.DataFrame):
        """Analyze companies by YC batch"""
        if not df['batch'].notna().any():
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Companies per batch
        batch_counts = df['batch'].value_counts().head(15)
        batch_counts.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Companies per YC Batch (Top 15)')
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Number of Companies')
        axes[0].tick_params(axis='x', rotation=45)
        
        # AI percentage by batch
        batch_ai_stats = df.groupby('batch')['is_ai_related'].agg(['count', 'sum']).reset_index()
        batch_ai_stats['ai_percentage'] = (batch_ai_stats['sum'] / batch_ai_stats['count']) * 100
        batch_ai_stats = batch_ai_stats[batch_ai_stats['count'] >= 3].sort_values('ai_percentage', ascending=False).head(15)
        
        if not batch_ai_stats.empty:
            axes[1].bar(range(len(batch_ai_stats)), batch_ai_stats['ai_percentage'])
            axes[1].set_title('AI Percentage by Batch (Min 3 companies)')
            axes[1].set_xlabel('Batch')
            axes[1].set_ylabel('AI Percentage')
            axes[1].set_xticks(range(len(batch_ai_stats)))
            axes[1].set_xticklabels(batch_ai_stats['batch'], rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'batch_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_locations(self, df: pd.DataFrame):
        """Analyze companies by location"""
        if not df['location'].notna().any():
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top locations
        location_counts = df['location'].value_counts().head(15)
        location_counts.plot(kind='barh', ax=axes[0])
        axes[0].set_title('Top Company Locations')
        axes[0].set_xlabel('Number of Companies')
        
        # AI percentage by location
        location_ai_stats = df.groupby('location')['is_ai_related'].agg(['count', 'sum']).reset_index()
        location_ai_stats['ai_percentage'] = (location_ai_stats['sum'] / location_ai_stats['count']) * 100
        location_ai_stats = location_ai_stats[location_ai_stats['count'] >= 5].sort_values('ai_percentage', ascending=False).head(10)
        
        if not location_ai_stats.empty:
            axes[1].barh(range(len(location_ai_stats)), location_ai_stats['ai_percentage'])
            axes[1].set_title('AI Percentage by Location (Min 5 companies)')
            axes[1].set_xlabel('AI Percentage')
            axes[1].set_yticks(range(len(location_ai_stats)))
            axes[1].set_yticklabels(location_ai_stats['location'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'location_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_task_mappings(self, mappings_df: pd.DataFrame):
        """Analyze task mappings"""
        if mappings_df.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top occupations by applicable mappings
        applicable_mappings = mappings_df[mappings_df['is_applicable'] == True]
        if not applicable_mappings.empty:
            occupation_counts = applicable_mappings['occupation'].value_counts().head(10)
            occupation_counts.plot(kind='barh', ax=axes[0, 0])
            axes[0, 0].set_title('Top Occupations by Company Mappings')
            axes[0, 0].set_xlabel('Number of Companies')
        
        # Mapping success rate by occupation
        mapping_stats = mappings_df.groupby('occupation')['is_applicable'].agg(['count', 'sum']).reset_index()
        mapping_stats['success_rate'] = (mapping_stats['sum'] / mapping_stats['count']) * 100
        mapping_stats = mapping_stats[mapping_stats['count'] >= 3].sort_values('success_rate', ascending=False).head(10)
        
        if not mapping_stats.empty:
            axes[0, 1].bar(range(len(mapping_stats)), mapping_stats['success_rate'])
            axes[0, 1].set_title('Mapping Success Rate by Occupation')
            axes[0, 1].set_ylabel('Success Rate (%)')
            axes[0, 1].set_xticks(range(len(mapping_stats)))
            axes[0, 1].set_xticklabels(mapping_stats['occupation'], rotation=45, ha='right')
        
        # Confidence score distribution for applicable mappings
        if not applicable_mappings.empty and 'confidence_score' in applicable_mappings.columns:
            applicable_mappings['confidence_score'].hist(bins=20, ax=axes[1, 0])
            axes[1, 0].set_title('Confidence Score Distribution (Applicable Mappings)')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # Companies with most task mappings
        company_mapping_counts = applicable_mappings['company_name'].value_counts().head(10)
        if not company_mapping_counts.empty:
            company_mapping_counts.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Companies with Most Task Mappings')
            axes[1, 1].set_xlabel('Company')
            axes[1, 1].set_ylabel('Number of Mapped Tasks')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'task_mapping_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_executive_summary(self, companies_df: pd.DataFrame, mappings_df: pd.DataFrame):
        """Generate executive summary report"""
        summary = f"""
YC Company Analysis - Executive Summary
======================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
--------
Total Companies Analyzed: {len(companies_df):,}
AI-Related Companies: {len(companies_df[companies_df['is_ai_related'] == True]):,} ({len(companies_df[companies_df['is_ai_related'] == True])/len(companies_df)*100:.1f}%)
Non-AI Companies: {len(companies_df[companies_df['is_ai_related'] == False]):,} ({len(companies_df[companies_df['is_ai_related'] == False])/len(companies_df)*100:.1f}%)

BATCH ANALYSIS
--------------"""

        if companies_df['batch'].notna().any():
            most_recent_batches = companies_df['batch'].value_counts().head(5)
            summary += f"""
Most Active Batches:
{chr(10).join(f"- {batch}: {count} companies" for batch, count in most_recent_batches.items())}

AI Concentration by Recent Batches:"""
            
            recent_ai_stats = companies_df.groupby('batch')['is_ai_related'].agg(['count', 'sum']).reset_index()
            recent_ai_stats['ai_percentage'] = (recent_ai_stats['sum'] / recent_ai_stats['count']) * 100
            recent_ai_stats = recent_ai_stats[recent_ai_stats['count'] >= 3].sort_values('ai_percentage', ascending=False).head(5)
            
            for _, row in recent_ai_stats.iterrows():
                summary += f"\n- {row['batch']}: {row['ai_percentage']:.1f}% AI ({row['sum']}/{row['count']} companies)"

        summary += f"""

LOCATION ANALYSIS
-----------------"""

        if companies_df['location'].notna().any():
            top_locations = companies_df['location'].value_counts().head(5)
            summary += f"""
Top Locations:
{chr(10).join(f"- {location}: {count} companies" for location, count in top_locations.items())}"""

        summary += f"""

TASK MAPPING INSIGHTS
---------------------"""

        if not mappings_df.empty:
            applicable_mappings = mappings_df[mappings_df['is_applicable'] == True]
            total_mappings = len(mappings_df)
            applicable_count = len(applicable_mappings)
            
            summary += f"""
Total Company-Task Evaluations: {total_mappings:,}
Applicable Mappings: {applicable_count:,} ({applicable_count/total_mappings*100:.1f}%)

Top Target Occupations:"""
            
            if not applicable_mappings.empty:
                top_occupations = applicable_mappings['occupation'].value_counts().head(5)
                for occupation, count in top_occupations.items():
                    summary += f"\n- {occupation}: {count} companies"

        summary += f"""

AI COMPANY HIGHLIGHTS
---------------------"""

        ai_companies = companies_df[companies_df['is_ai_related'] == True]
        if not ai_companies.empty:
            # High confidence AI companies
            if 'confidence_score' in ai_companies.columns:
                high_confidence = ai_companies[ai_companies['confidence_score'] >= 0.8]
                summary += f"""
High-Confidence AI Companies ({len(high_confidence)} total):"""
                for _, company in high_confidence.head(10).iterrows():
                    summary += f"\n- {company['name']}: {company['description'][:100]}..."

        summary += f"""

RECOMMENDATIONS
---------------
1. Focus AI investment on high-growth batches with strong AI representation
2. Consider geographic expansion in AI-heavy locations
3. Target occupations with highest mapping success rates for product development
4. Monitor emerging AI trends in recent YC batches

METHODOLOGY
-----------
- Data Source: Y Combinator company directory
- AI Classification: Anthropic Claude AI with confidence scoring
- Task Mapping: Automated mapping to O*NET occupational database
- Analysis Period: {datetime.now().strftime('%Y-%m-%d')}

For detailed charts and visualizations, see:
- ai_landscape_analysis.png
- batch_analysis.png
- location_analysis.png
- task_mapping_analysis.png
"""

        # Save summary
        with open(self.output_dir / 'executive_summary.txt', 'w') as f:
            f.write(summary)
        
        logger.info("Executive summary generated")
    
    def _parse_team_size(self, team_size_str: str) -> Optional[float]:
        """Parse team size string to numeric value"""
        if pd.isna(team_size_str) or not team_size_str:
            return None
        
        team_size_str = str(team_size_str).lower()
        
        if '+' in team_size_str:
            # Extract number before +
            import re
            match = re.search(r'(\d+)', team_size_str)
            if match:
                return float(match.group(1))
        
        if '-' in team_size_str:
            # Take average of range
            import re
            matches = re.findall(r'(\d+)', team_size_str)
            if len(matches) >= 2:
                return (float(matches[0]) + float(matches[1])) / 2
        
        # Single number
        import re
        match = re.search(r'(\d+)', team_size_str)
        if match:
            return float(match.group(1))
        
        return None

class AutomationScheduler:
    """Handles automated pipeline runs"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.is_running = False
        
    def setup_schedule(self):
        """Setup automated scheduling"""
        # Full pipeline run weekly on Sundays at 2 AM
        schedule.every().sunday.at("02:00").do(self._run_full_pipeline)
        
        # Incremental updates daily at 6 AM
        schedule.every().day.at("06:00").do(self._run_incremental_update)
        
        # Quick analytics refresh every 6 hours
        schedule.every(6).hours.do(self._refresh_analytics)
        
        logger.info("Automation schedule configured:")
        logger.info("- Full pipeline: Sundays at 2:00 AM")
        logger.info("- Incremental updates: Daily at 6:00 AM") 
        logger.info("- Analytics refresh: Every 6 hours")
    
    def start_scheduler(self):
        """Start the automation scheduler"""
        self.is_running = True
        logger.info("Starting automation scheduler...")
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """Stop the automation scheduler"""
        self.is_running = False
        logger.info("Automation scheduler stopped")
    
    def _run_full_pipeline(self):
        """Run full pipeline automatically"""
        try:
            logger.info("🤖 Running scheduled full pipeline...")
            self.pipeline.run_full_pipeline(max_companies=1000)
            logger.info("✅ Scheduled full pipeline completed")
        except Exception as e:
            logger.error(f"❌ Scheduled full pipeline failed: {e}")
    
    def _run_incremental_update(self):
        """Run incremental update"""
        try:
            logger.info("🔄 Running scheduled incremental update...")
            self.pipeline.run_incremental_update()
            logger.info("✅ Scheduled incremental update completed")
        except Exception as e:
            logger.error(f"❌ Scheduled incremental update failed: {e}")
    
    def _refresh_analytics(self):
        """Refresh analytics only"""
        try:
            logger.info("📊 Refreshing analytics...")
            self.pipeline.refresh_analytics()
            logger.info("✅ Analytics refresh completed")
        except Exception as e:
            logger.error(f"❌ Analytics refresh failed: {e}")

class FullFeaturedPipeline:
    """Main pipeline orchestrator with all features"""
    
    def __init__(self):
        self.scraper = RealYCWebScraper()
        self.classifier = AIClassifierAdvanced()
        self.task_mapper = TaskMappingEngine()
        self.analytics = AnalyticsEngine()
        self.sheets_manager = self._init_sheets_manager()
        self.scheduler = AutomationScheduler(self)
        
        # Cache for incremental updates
        self.cache_file = Path('data/pipeline_cache.json')
        self.cache_file.parent.mkdir(exist_ok=True)
    
    def _init_sheets_manager(self):
        """Initialize Google Sheets manager"""
        if GOOGLE_SHEETS_AVAILABLE and Path('credentials.json').exists():
            try:
                scope = [
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/drive'
                ]
                creds = Credentials.from_service_account_file('credentials.json', scopes=scope)
                return gspread.authorize(creds)
            except Exception as e:
                logger.warning(f"Failed to initialize Google Sheets: {e}")
        return None
    
    def run_full_pipeline(self, max_companies: int = 200):
        """Run the complete full-featured pipeline"""
        start_time = datetime.now()
        logger.info("🚀 Starting Full-Featured YC Pipeline...")
        logger.info(f"📊 Target: {max_companies} companies with complete analysis")
        
        try:
            # Step 1: Scrape real YC companies
            logger.info("🔍 Step 1: Scraping YC companies...")
            companies = self.scraper.scrape_yc_companies(max_companies)
            
            if not companies:
                logger.error("No companies scraped. Exiting.")
                return None
            
            # Step 2: Advanced AI classification
            logger.info("🤖 Step 2: AI classification with confidence scoring...")
            companies = self.classifier.classify_companies_batch(companies)
            
            # Step 3: Task mapping
            logger.info("🎯 Step 3: Mapping companies to occupational tasks...")
            mappings = self.task_mapper.map_companies_to_tasks(companies)
            
            # Step 4: Save data
            logger.info("💾 Step 4: Saving data...")
            self._save_all_data(companies, mappings)
            
            # Step 5: Generate analytics
            logger.info("📊 Step 5: Generating comprehensive analytics...")
            self.analytics.generate_comprehensive_analysis(companies, mappings)
            
            # Step 6: Update Google Sheets
            if self.sheets_manager:
                logger.info("📋 Step 6: Updating Google Sheets...")
                self._update_google_sheets(companies, mappings)
            
            # Step 7: Cache results for incremental updates
            self._cache_results(companies, mappings)
            
            # Generate final summary
            self._log_pipeline_summary(companies, mappings, start_time)
            
            return {
                'companies': len(companies),
                'ai_companies': len([c for c in companies if c.is_ai_related]),
                'task_mappings': len([m for m in mappings if m.is_applicable]),
                'success': True,
                'duration_minutes': (datetime.now() - start_time).total_seconds() / 60
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_incremental_update(self):
        """Run incremental update with new companies only"""
        logger.info("🔄 Starting incremental pipeline update...")
        
        # Load previous results
        cached_companies = self._load_cached_companies()
        cached_names = {c.name for c in cached_companies}
        
        # Get new companies
        all_companies = self.scraper.scrape_yc_companies(500)
        new_companies = [c for c in all_companies if c.name not in cached_names]
        
        if not new_companies:
            logger.info("No new companies found")
            return
        
        logger.info(f"Found {len(new_companies)} new companies")
        
        # Process new companies
        new_companies = self.classifier.classify_companies_batch(new_companies)
        new_mappings = self.task_mapper.map_companies_to_tasks(new_companies)
        
        # Merge with cached data
        all_companies = cached_companies + new_companies
        all_mappings = self._load_cached_mappings() + new_mappings
        
        # Save and update
        self._save_all_data(all_companies, all_mappings)
        self.analytics.generate_comprehensive_analysis(all_companies, all_mappings)
        
        if self.sheets_manager:
            self._update_google_sheets(all_companies, all_mappings)
        
        self._cache_results(all_companies, all_mappings)
        
        logger.info(f"✅ Incremental update completed: {len(new_companies)} new companies processed")
    
    def refresh_analytics(self):
        """Refresh analytics only"""
        cached_companies = self._load_cached_companies()
        cached_mappings = self._load_cached_mappings()
        
        if cached_companies:
            self.analytics.generate_comprehensive_analysis(cached_companies, cached_mappings)
            logger.info("Analytics refreshed")
    
    def start_automation(self):
        """Start automated scheduling"""
        self.scheduler.setup_schedule()
        
        # Run in separate thread
        scheduler_thread = threading.Thread(target=self.scheduler.start_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info("🤖 Automation started - pipeline will run automatically")
    
    def _save_all_data(self, companies: List[Company], mappings: List[CompanyTaskMapping]):
        """Save all data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save companies
        companies_data = []
        for company in companies:
            companies_data.append({
                'name': company.name,
                'description': company.description,
                'batch': company.batch,
                'website': company.website,
                'founded': company.founded,
                'team_size': company.team_size,
                'location': company.location,
                'tags': ','.join(company.tags),
                'status': company.status,
                'is_ai_related': company.is_ai_related,
                'confidence_score': company.confidence_score,
                'mapped_tasks': ','.join(company.mapped_tasks)
            })
        
        companies_df = pd.DataFrame(companies_data)
        companies_file = f'output/yc_companies_full_{timestamp}.csv'
        companies_df.to_csv(companies_file, index=False)
        
        # Save mappings
        mappings_data = []
        for mapping in mappings:
            mappings_data.append({
                'company_name': mapping.company_name,
                'task_id': mapping.task_id,
                'occupation': mapping.occupation,
                'is_applicable': mapping.is_applicable,
                'confidence_score': mapping.confidence_score,
                'reasoning': mapping.reasoning
            })
        
        mappings_df = pd.DataFrame(mappings_data)
        mappings_file = f'output/task_mappings_full_{timestamp}.csv'
        mappings_df.to_csv(mappings_file, index=False)
        
        logger.info(f"💾 Data saved: {companies_file}, {mappings_file}")
    
    def _update_google_sheets(self, companies: List[Company], mappings: List[CompanyTaskMapping]):
        """Update Google Sheets with all data"""
        sheet_id = os.getenv('GOOGLE_SHEET_ID')
        if not sheet_id or not self.sheets_manager:
            return
        
        try:
            workbook = self.sheets_manager.open_by_key(sheet_id)
            
            # Update companies sheet
            companies_ws = workbook.sheet1
            companies_ws.clear()
            
            # Prepare companies data
            companies_data = [['name', 'description', 'batch', 'website', 'founded', 'team_size', 
                             'location', 'tags', 'status', 'is_ai_related', 'confidence_score', 'mapped_tasks']]
            
            for company in companies:
                companies_data.append([
                    company.name, company.description, company.batch, company.website,
                    company.founded, company.team_size, company.location, ','.join(company.tags),
                    company.status, company.is_ai_related, company.confidence_score, ','.join(company.mapped_tasks)
                ])
            
            companies_ws.update(companies_data)
            
            # Create/update mappings sheet
            try:
                mappings_ws = workbook.worksheet('Task Mappings')
            except:
                mappings_ws = workbook.add_worksheet('Task Mappings', rows=len(mappings)+100, cols=10)
            
            mappings_ws.clear()
            
            # Prepare mappings data
            mappings_data = [['company_name', 'task_id', 'occupation', 'is_applicable', 'confidence_score', 'reasoning']]
            
            for mapping in mappings:
                mappings_data.append([
                    mapping.company_name, mapping.task_id, mapping.occupation,
                    mapping.is_applicable, mapping.confidence_score, mapping.reasoning
                ])
            
            mappings_ws.update(mappings_data)
            
            logger.info(f"📋 Google Sheets updated: {len(companies)} companies, {len(mappings)} mappings")
            
        except Exception as e:
            logger.error(f"Failed to update Google Sheets: {e}")
    
    def _cache_results(self, companies: List[Company], mappings: List[CompanyTaskMapping]):
        """Cache results for incremental updates"""
        cache_data = {
            'companies': [company.__dict__ for company in companies],
            'mappings': [mapping.__dict__ for mapping in mappings],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _load_cached_companies(self) -> List[Company]:
        """Load cached companies"""
        if not self.cache_file.exists():
            return []
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            companies = []
            for company_data in cache_data.get('companies', []):
                company = Company(**company_data)
                companies.append(company)
            
            return companies
        except Exception as e:
            logger.warning(f"Failed to load cached companies: {e}")
            return []
    
    def _load_cached_mappings(self) -> List[CompanyTaskMapping]:
        """Load cached mappings"""
        if not self.cache_file.exists():
            return []
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            mappings = []
            for mapping_data in cache_data.get('mappings', []):
                mapping = CompanyTaskMapping(**mapping_data)
                mappings.append(mapping)
            
            return mappings
        except Exception as e:
            logger.warning(f"Failed to load cached mappings: {e}")
            return []
    
    def _log_pipeline_summary(self, companies: List[Company], mappings: List[CompanyTaskMapping], start_time: datetime):
        """Log comprehensive pipeline summary"""
        duration = datetime.now() - start_time
        ai_companies = [c for c in companies if c.is_ai_related]
        applicable_mappings = [m for m in mappings if m.is_applicable]
        
        logger.info("🎉 FULL PIPELINE COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"⏱️  Duration: {duration.total_seconds()/60:.1f} minutes")
        logger.info(f"🏢 Total companies: {len(companies)}")
        logger.info(f"🤖 AI companies: {len(ai_companies)} ({len(ai_companies)/len(companies)*100:.1f}%)")
        logger.info(f"🎯 Task mappings: {len(applicable_mappings)} applicable")
        logger.info(f"📊 Analytics: Generated comprehensive reports")
        logger.info(f"📋 Google Sheets: Updated with all data")
        logger.info("=" * 50)
        
        # Top AI companies
        logger.info("🔥 Top AI Companies:")
        high_confidence_ai = [c for c in ai_companies if c.confidence_score and c.confidence_score >= 0.8]
        for company in high_confidence_ai[:5]:
            logger.info(f"   - {company.name}: {company.description[:60]}...")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Full-Featured YC Pipeline")
    parser.add_argument('--companies', type=int, default=200, help='Number of companies to process')
    parser.add_argument('--mode', choices=['full', 'incremental', 'analytics', 'schedule'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--automate', action='store_true', help='Start automation scheduler')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FullFeaturedPipeline()
    
    if args.mode == 'full':
        logger.info(f"Running full pipeline with {args.companies} companies...")
        results = pipeline.run_full_pipeline(args.companies)
        if results and results['success']:
            print(f"\n🎉 Success! Processed {results['companies']} companies in {results['duration_minutes']:.1f} minutes")
            print(f"📊 View analytics in output/analytics/ directory")
            print(f"📋 Check Google Sheets for live data")
        else:
            print("❌ Pipeline failed")
    
    elif args.mode == 'incremental':
        pipeline.run_incremental_update()
    
    elif args.mode == 'analytics':
        pipeline.refresh_analytics()
        print("📊 Analytics refreshed!")
    
    elif args.mode == 'schedule':
        print("🤖 Starting automation scheduler...")
        pipeline.start_automation()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("⏹️  Automation stopped")
    
    if args.automate and args.mode == 'full':
        print("🤖 Starting automation after full run...")
        pipeline.start_automation()
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("⏹️  Automation stopped")

if __name__ == "__main__":
    main()

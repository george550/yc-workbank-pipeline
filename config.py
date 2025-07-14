# Save this as: config.py
#!/usr/bin/env python3
"""
Configuration management for the YC-WORKBank pipeline
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for pipeline settings"""
    
    # API Configuration
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'anthropic')
    
    # Data Configuration
    WORKBANK_FILE = os.getenv('WORKBANK_FILE', 'data/workbank_tasks.csv')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
    
    # Rate Limiting
    API_RATE_LIMIT_DELAY = float(os.getenv('API_RATE_LIMIT_DELAY', '1.0'))
    SCRAPING_DELAY = float(os.getenv('SCRAPING_DELAY', '0.5'))
    
    @classmethod
    def to_dict(cls):
        """Convert configuration to dictionary"""
        return {
            'llm_provider': cls.LLM_PROVIDER,
            'workbank_file': cls.WORKBANK_FILE,
            'output_dir': cls.OUTPUT_DIR,
            'api_rate_limit_delay': cls.API_RATE_LIMIT_DELAY,
            'scraping_delay': cls.SCRAPING_DELAY
        }

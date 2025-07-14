companies = [
            {"name": "OpenAI", "description": "AI research company developing large language models and AI systems", "is_ai_related": None},
            {"name": "Anthropic", "description": "AI safety company focused on developing safe, beneficial AI systems", "is_ai_related": None},
            {"name": "Stripe", "description": "Online payment processing platform for businesses", "is_ai_related": None},
            {"name": "Airbnb", "description": "Online marketplace for lodging and travel experiences", "is_ai_related": None},
            {"name": "Dropbox", "description": "Cloud storage and file synchronization service", "is_ai_related": None},
            {"name": "Instacart", "description": "Grocery delivery and pickup service", "is_ai_related": None},
            {"name": "DoorDash", "description": "Food delivery platform connecting customers with restaurants", "is_ai_related": None},
            {"name": "Coinbase", "description": "Cryptocurrency exchange and wallet platform", "is_ai_related": None},
            {"name": "Reddit", "description": "Social news aggregation and discussion website", "is_ai_related": None},
            {"name": "Zapier", "description": "Automation platform connecting different apps and services", "is_ai_related": None},
            {"name": "Scale AI", "description": "Data platform for AI providing training data and model evaluation", "is_ai_related": None},
            {"name": "Hugging Face", "description": "Platform for machine learning models and datasets", "is_ai_related": None},
            {"name": "Weights & Biases", "description": "MLOps platform for experiment tracking and model management", "is_ai_related": None},
            {"name": "Twilio", "description": "Cloud communications platform for messaging and voice", "is_ai_related": None},
            {"name": "Notion", "description": "Productivity software for note-taking and project management", "is_ai_related": None},
        ]#!/usr/bin/env python3
"""Simple YC-WORKBank Pipeline for testing"""

import csv
import time
import logging
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YCWorkBankPipeline:
    def __init__(self):
        pass
    
    def run_pipeline(self):
        """Run a simple test pipeline"""
        logger.info("🚀 Starting YC-WORKBank Pipeline...")
        
        # Sample companies
        companies = [
            {"name": "OpenAI", "description": "AI research company", "is_ai_related": True},
            {"name": "Stripe", "description": "Payment processing", "is_ai_related": False},
            {"name": "Anthropic", "description": "AI safety company", "is_ai_related": True},
        ]
        
        # Save results
        os.makedirs("output", exist_ok=True)
        with open("output/yc_companies.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "description", "is_ai_related"])
            writer.writeheader()
            writer.writerows(companies)
        
        logger.info("✅ Pipeline completed successfully!")
        return {"total_companies": len(companies), "success": True}

def main():
    pipeline = YCWorkBankPipeline()
    results = pipeline.run_pipeline()
    print(f"Results: {results}")

if __name__ == "__main__":
    main()


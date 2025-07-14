#!/usr/bin/env python3
"""Simple CLI for YC-WORKBank pipeline"""

import argparse
from yc_workbank_pipeline import YCWorkBankPipeline

def run_pipeline(args):
    print("🚀 Running YC-WORKBank Pipeline...")
    pipeline = YCWorkBankPipeline()
    results = pipeline.run_pipeline()
    print(f"✅ Completed: {results}")

def main():
    parser = argparse.ArgumentParser(description="YC-WORKBank Pipeline")
    subparsers = parser.add_subparsers(dest='command')
    
    run_parser = subparsers.add_parser('run', help='Run the pipeline')
    run_parser.set_defaults(func=run_pipeline)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


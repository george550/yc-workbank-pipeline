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

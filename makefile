.PHONY: quality style unit-test integ-test

check_dirs := src 

# run tests

unit-test:
	python3 -m pytest -s -v ./tests/unit

integ-test:
	python3 -m pytest -s -v ./tests/integ/

# Check that source code meets quality standards

quality:
	ruff $(check_dirs) 

# Format source code automatically

style: 
	ruff $(check_dirs) --fix
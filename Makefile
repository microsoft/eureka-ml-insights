PATH  := $(PATH)
SHELL := /bin/bash
LINE_LENGTH := 120
MULTI_LINE := 3
flake:
	flake8 -v --max-line-length $(LINE_LENGTH) --per-file-ignores="tests/*:E402,E501" --exclude "eureka_ml_insights/configs" ./
isort:
	isort --check-only --multi-line $(MULTI_LINE) --trailing-comma --diff ./
black:
	black --check --line-length $(LINE_LENGTH) ./
linters:
	make isort; \
	make black; \
	make flake; 
isort-inplace:
	isort --multi-line $(MULTI_LINE) --trailing-comma ./
black-inplace:
	black --line-length $(LINE_LENGTH) ./
autoflake-inplace:
	autoflake --remove-all-unused-imports --in-place --remove-unused-variables -r ./
format-inplace:
	make autoflake-inplace
	make isort-inplace
	make black-inplace
test:
	python run_tests.py

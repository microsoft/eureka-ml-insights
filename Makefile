PATH  := $(PATH)
SHELL := /bin/bash
LINE_LENGTH := 120
MULTI_LINE := 3
MERGE_BASE := $(shell git merge-base origin/main HEAD)
DIFF := $(shell git diff --name-only --diff-filter=db $(MERGE_BASE) | grep -E '\.py$$')
flake:
	flake8 -v --max-line-length $(LINE_LENGTH) --per-file-ignores="tests/*:E402,E501" --exclude "eureka_ml_insights/configs" $(DIFF)
isort:
	isort --check-only --multi-line $(MULTI_LINE) --trailing-comma --diff $(DIFF)
black:
	black --check --line-length $(LINE_LENGTH) $(DIFF)
linters:
	@if [ -z "$(DIFF)" ]; then \
		echo "No files to format."; \
	else \
		make isort; \
		make black; \
		make flake; \
	fi
isort-inplace:
	isort --multi-line $(MULTI_LINE) --trailing-comma $(DIFF)
black-inplace:
	black --line-length $(LINE_LENGTH) $(DIFF)
autoflake-inplace:
	autoflake --remove-all-unused-imports --in-place --remove-unused-variables -r $(DIFF)
format-inplace:
	make autoflake-inplace
	make isort-inplace
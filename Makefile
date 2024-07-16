.PHONY: help
help:			## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets: "
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: clean
clean:			## Clean unused files.
	@echo "Cleaning up..."
	@find . -name "*.egg-info" -exec rm -rf {} +
	@find . -name "*.ipynb_checkpoints" -exec rm -rf {} +
	@find . -name "*.pyc" -exec rm -rf {} +
	@find . -name "__pycache__" -exec rm -rf {} +
	@rm -f .coverage
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@rm -rf xrkit/*.egg-info
	@rm -rf htmlcov
	@rm -rf docs/_build
	@rm -rf docs/_static


.PHONY: install
install:		## Install in development mode.
	pip install -e .[test]


.PHONY: format
format:			## Format code using isort and black
	isort .
	black -l 110 .


.PHONY: lint
lint:			## Run linters
	flake8 xrkit/
	black -l 110 --check .
	mypy xrkit/


.PHONY: test
test: lint		## Run tests and generate coverage report
	pytest tests/
	coverage html


.PHONY: docs
docs:			## Build documentation
	@echo "Building documentation..."
	pdoc xrkit -o docs
	@echo "Serving API documentation..." 
	pdoc xrkit --host localhost --port 8080
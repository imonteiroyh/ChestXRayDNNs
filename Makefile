.PHONY: help
help:			## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets: "
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: clean
clean:			## Clean unused files.
	@echo "Cleaning up..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
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
	isort notebooks/
	isort scripts/
	isort xrkit/
	isort tests/
	black -l 110 notebooks/
	black -l 110 scripts/
	black -l 110 xrkit/
	black -l 110 tests/


.PHONY: lint
lint:			## Run linters
	flake8 xrkit/
	black -l 110 --check xrkit/
	black -l 110 --check tests/
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

.DEFAULT_GOAL := help

.PHONY: requirements

requirements: ## Install requirements
	pip install -r requirements/base.txt

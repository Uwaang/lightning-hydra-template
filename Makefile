UV ?= uv
ARGS ?=

.PHONY: help install install-all format lint test test-full train eval clean clean-logs docker-build docker-build-core docker-train

help: ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Sync the locked core environment
	$(UV) sync

install-all: ## Sync the locked environment with every optional extra
	$(UV) sync --all-extras

format: ## Run all pre-commit hooks and apply safe fixes
	$(UV) run pre-commit run --all-files

lint: ## Check Python lint and formatting without modifying files
	$(UV) run ruff check .
	$(UV) run ruff format --check .

test: ## Run the non-slow test suite
	$(UV) run python -m pytest -m "not slow"

test-full: ## Run all tests, including slow tests
	$(UV) run python -m pytest

train: ## Train; append Hydra overrides with ARGS="..."
	$(UV) run train-command $(ARGS)

eval: ## Evaluate; pass ckpt_path and overrides with ARGS="..."
	$(UV) run eval-command $(ARGS)

clean: ## Clean generated Python and test artifacts
	rm -rf dist .pytest_cache htmlcov
	rm -f .coverage coverage.xml
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +

clean-logs: ## Clean Hydra and Lightning logs
	rm -rf logs/*

docker-build: ## Build the full CUDA development image
	docker build -t lightning-hydra:dev .

docker-build-core: ## Build a core-only CUDA image
	docker build --build-arg INSTALL_OPTIONAL=false -t lightning-hydra:core .

docker-train: ## Train in Docker with all NVIDIA GPUs
	docker run --rm --gpus all --ipc=host -v "$(PWD)/data:/workspace/data" -v "$(PWD)/logs:/workspace/logs" lightning-hydra:dev python src/train.py trainer=gpu

# Makefile for BitNet-7B-KDE

# === Core Variables ===
PYTHON := python3
PY := $(PYTHON)
PIP := $(PYTHON) -m pip
SHELL := /bin/bash
.DEFAULT_GOAL := help

# === Colors for Output ===
NC := \033[0m
RED := \033[0;31m
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m

# === Helper Macros ===
define LOG
	@echo -e "$(BLUE)[BitNet-7B-KDE]$(NC) $(1)"
endef

define SUCCESS
	@echo -e "$(GREEN)✓$(NC) $(1)"
endef

define WARN
	@echo -e "$(YELLOW)⚠$(NC) $(1)"
endef

define ERROR
	@echo -e "$(RED)✗$(NC) $(1)"
endef

define RUN
	@echo -e "$(BLUE)▶$(NC) Running: $(1)"
	@$(1)
endef

# === Targets ===

.PHONY: help
help:  ## Show this help message
	@echo -e "$(BLUE)BitNet-7B-KDE Makefile$(NC)"
	@echo -e "$(GREEN)Usage:$(NC) make [target]"
	@echo ""
	@echo -e "$(YELLOW)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}'

.PHONY: install
install:  ## Install all dependencies
	$(call LOG,"Installing dependencies...")
	@$(PIP) install -r requirements.txt
	$(call SUCCESS,"Dependencies installed")

.PHONY: install-dev
install-dev:  ## Install dev dependencies
	$(call LOG,"Installing dev dependencies...")
	@$(PIP) install pytest black isort mypy
	$(call SUCCESS,"Dev dependencies installed")

.PHONY: env
env:  ## Create .env from template
	$(call LOG,"Setting up environment...")
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓$(NC) Created .env from template"; \
		echo "$(YELLOW)⚠$(NC) Please edit .env with your API keys"; \
	else \
		echo "$(BLUE)ℹ$(NC) .env already exists"; \
	fi

.PHONY: ensure_dirs
ensure_dirs:  ## Create required directories
	$(call LOG,"Creating directory structure...")
	@mkdir -p data/{teacher,kd,eval}
	@mkdir -p checkpoints
	@mkdir -p logs
	@mkdir -p outputs
	$(call SUCCESS,"Directories created")

.PHONY: clean
clean:  ## Clean generated files
	$(call LOG,"Cleaning generated files...")
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name ".DS_Store" -delete
	$(call SUCCESS,"Cleaned")

.PHONY: prompts
prompts: ensure_dirs  ## Generate test prompts
	$(call LOG,"Generating test prompts...")
	@$(call RUN, $(PY) -m scripts.generate_prompts --count 25)
	$(call SUCCESS,"Prompts generated in data/prompts.json")

.PHONY: sanity
sanity: ensure_dirs  ## Run sanity check
	$(call LOG,"Running sanity check...")
	@$(call RUN, $(PY) -m scripts.sanity_check)

.PHONY: teacher
teacher: ensure_dirs  ## Collect teacher baseline samples
	$(call LOG,"Collecting teacher baseline samples...")
	@$(call RUN, $(PY) -m scripts.run_teacher_baseline)

.PHONY: collect
collect: ensure_dirs  ## Collect KD traces from teacher
	$(call LOG,"Collecting KD traces...")
	@$(call RUN, $(PY) -m scripts.collect_kd_traces)

.PHONY: train
train: ensure_dirs  ## Train mini BitNet model
	$(call LOG,"Training mini BitNet model...")
	@$(call RUN, $(PY) -m scripts.train_mini_bitnet)

.PHONY: eval
eval: ensure_dirs  ## Evaluate model and compute QEI
	$(call LOG,"Evaluating model...")
	@$(call RUN, $(PY) -m scripts.eval_and_qei)

.PHONY: dryrun
dryrun: ensure_dirs  ## Dry run 7B model memory test
	$(call LOG,"Running 7B memory dry run...")
	@$(call RUN, $(PY) -m scripts.dry_run_7b_memory)

.PHONY: test
test:  ## Run tests
	$(call LOG,"Running tests...")
	@$(PY) -m pytest tests/ -v
	$(call SUCCESS,"Tests completed")

.PHONY: format
format:  ## Format code with black and isort
	$(call LOG,"Formatting code...")
	@black src/ scripts/ tests/ --line-length 120
	@isort src/ scripts/ tests/ --profile black
	$(call SUCCESS,"Code formatted")

.PHONY: lint
lint:  ## Run linting checks
	$(call LOG,"Running linters...")
	@black src/ scripts/ tests/ --check --line-length 120
	@isort src/ scripts/ tests/ --check --profile black
	$(call SUCCESS,"Linting passed")

.PHONY: notebook
notebook:  ## Open Jupyter notebook
	$(call LOG,"Starting Jupyter notebook...")
	@jupyter notebook notebooks/

.PHONY: pipeline
pipeline: ensure_dirs teacher collect train eval  ## Run full pipeline
	$(call LOG,"Pipeline complete!")
	$(call SUCCESS,"All steps executed successfully")

.PHONY: quick
quick: ensure_dirs train eval  ## Quick train+eval (assumes data exists)
	$(call LOG,"Quick pipeline complete!")
	$(call SUCCESS,"Training and evaluation done")

.PHONY: all
all: install env ensure_dirs pipeline  ## Install and run everything
	$(call LOG,"Full setup and pipeline complete!")
	$(call SUCCESS,"BitNet-7B-KDE ready!")

# === Development Helpers ===

.PHONY: watch
watch:  ## Watch training logs
	@tail -f logs/train_*.log 2>/dev/null || echo "No training logs found"

.PHONY: gpu
gpu:  ## Show GPU status
	@nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"

.PHONY: disk
disk:  ## Show disk usage of data directories
	@du -sh data/* checkpoints/* 2>/dev/null | sort -h

.PHONY: count
count:  ## Count lines of code
	@echo "Lines of code in src/:"
	@find src -name "*.py" | xargs wc -l | tail -1
	@echo "Lines of code in scripts/:"
	@find scripts -name "*.py" | xargs wc -l | tail -1

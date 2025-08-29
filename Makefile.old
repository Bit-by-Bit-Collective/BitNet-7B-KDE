# Makefile — BitNet-7B-KDE
# Common targets: teacher, collect, train, eval, dryrun
# Usage:
#   make install
#   make teacher
#   make collect
#   make train
#   make eval
#   make dryrun
# You can override .env vars on the CLI, e.g.:
#   make train TOTAL_STEPS=200 KD_TAU=1.1

SHELL := /usr/bin/env bash
PY ?= python3
ENV_FILE ?= .env
.EXPORT_ALL_VARIABLES:

# Small helper to run a command with .env sourced into the environment.
define RUN
bash -lc 'set -euo pipefail; set -a; [ -f $(ENV_FILE) ] && source $(ENV_FILE); set +a; $(1)'
endef

.PHONY: help install env check_env ensure_dirs teacher collect train eval dryrun clean

help:
	@echo "Targets:"
	@echo "  install     - pip install requirements.txt"
	@echo "  teacher     - run deterministic teacher baseline"
	@echo "  collect     - collect KD traces (Top-K + Other) to Parquet"
	@echo "  train       - train the mini BitNet with KD+CE+format losses"
	@echo "  eval        - run quick eval + QEI report"
	@echo "  dryrun      - 7B forward-pass memory dry-run"
	@echo "  ensure_dirs - mount & create storage dirs from .env"
	@echo "  check_env   - validate key env vars"
	@echo "  clean       - remove transient cache dirs (safe)"

install:
	@$(call RUN, $(PY) -m pip install -U pip && $(PY) -m pip install -r requirements.txt)

env:
	@if [ ! -f "$(ENV_FILE)" ]; then \
	  echo "ERROR: $(ENV_FILE) not found. Copy .env.example to .env and edit values."; \
	  exit 1; \
	fi
	@echo "Using env file: $(ENV_FILE)"

check_env: env
	@$(call RUN, \
		for v in DRIVE_ROOT TOKENIZER_NAME PROVIDER; do \
			if [ -z "$${!v:-}" ]; then echo "Missing $$v in $(ENV_FILE)"; exit 1; fi; \
		done; \
		echo "✓ .env looks OK." \
	)

ensure_dirs: env
	@$(call RUN, \
		'$(PY)' - <<'PY' \
from scripts.storage import prepare_storage; prepare_storage(verbose=True) \
PY \
	)

teacher: ensure_dirs
	@$(call RUN, $(PY) -m scripts.teacher_baseline)

collect: ensure_dirs
	@$(call RUN, $(PY) -m scripts.collect_kd)

train: ensure_dirs
	@$(call RUN, $(PY) -m scripts.train)

eval: ensure_dirs
	@$(call RUN, $(PY) -m scripts.eval)

dryrun: ensure_dirs
	@$(call RUN, $(PY) -m scripts.dryrun_7b)

clean:
	@$(call RUN, \
		rm -rf "$${TRANSFORMERS_CACHE:-}" "$${HF_DATASETS_CACHE:-}" "$${TORCH_HOME:-}"; \
		echo "✓ cleaned caches (if set in .env)." \
	)

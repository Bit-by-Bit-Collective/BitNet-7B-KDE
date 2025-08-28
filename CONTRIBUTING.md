# Contributing to BitNet-7B-KDE

## Dev setup
- Python 3.10+
- `cp .env.example .env` and fill in keys/paths
- `make install && make ensure_dirs`

## Common tasks
- Teacher baseline: `make teacher`
- KD collection: `make collect`
- Train mini: `make train`
- Eval + QEI: `make eval`
- 7B dry-run: `make dryrun`

## Style
- Black + Ruff (`make lint` upcoming)
- Type hints preferred
- Small, focused PRs with before/after notes

## Tests
- Unit tests (losses, projection, masks) – `pytest -q` (coming soon)

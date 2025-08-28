# Release Process

We follow SemVer: MAJOR.MINOR.PATCH

## Before Release
- Update `CHANGELOG.md`.
- Ensure `.env.example` matches latest env usage.
- CI green on:
  - lint/tests (once added),
  - `make teacher`, `make collect`, `make train`, `make eval`, `make dryrun` smoke.

## Tagging

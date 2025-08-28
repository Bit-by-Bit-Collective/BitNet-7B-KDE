# Storage & Paths

This project is **.env-driven**. All scripts call `scripts/storage.py:prepare_storage()` which:
- (Optionally) mounts Google Drive in Colab,
- Ensures folders exist,
- Exports cache envs.

## Default (Google Drive in Colab)

Required in `.env`:

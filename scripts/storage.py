# scripts/storage.py
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False

def _mount_gdrive(mount_point: str):
    from google.colab import drive  # type: ignore
    drive.mount(mount_point)

def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name, str(int(default))).strip().lower()
    return val in ("1", "true", "yes", "y", "on")

def _required(name: str) -> str:
    v = os.getenv(name)
    if not v or not v.strip():
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def prepare_storage(verbose: bool = True) -> dict:
    """
    Mounts Google Drive (if requested) and ensures all folders exist,
    using ONLY values supplied via .env.
    Returns resolved paths.
    """
    # --- REQUIRED (must be in .env) ---
    # Root project folder on Drive (e.g. /content/drive/MyDrive/bitnet_poc)
    drive_root = _required("DRIVE_ROOT")

    # --- OPTIONAL (in .env, with sensible defaults) ---
    auto_mount = _bool_env("AUTO_MOUNT_GDRIVE", True)
    mount_point = os.getenv("GDRIVE_MOUNT_POINT", "/content/drive")
    # If DRIVE_ROOT starts with your mount point, we can mount
    should_mount = _in_colab() and auto_mount and drive_root.startswith(mount_point)

    if should_mount:
        base = Path(mount_point)
        # If mount folder exists but looks empty, try mounting
        try:
            empty = (not base.exists()) or (base.exists() and not any(base.iterdir()))
        except Exception:
            empty = True
        if empty:
            if verbose:
                print(f"🔧 Mounting Google Drive at {mount_point}...")
            _mount_gdrive(mount_point)

    # Resolve core folders (all from .env, or fallback to under DRIVE_ROOT)
    checkpoints = os.getenv("CHECKPOINTS_DIR", str(Path(drive_root) / "checkpoints"))
    data = os.getenv("DATA_DIR", str(Path(drive_root) / "data"))
    reports = os.getenv("REPORTS_DIR", str(Path(drive_root) / "reports"))
    logs = os.getenv("LOGS_DIR", str(Path(drive_root) / "logs"))

    # Caches (explicit from .env or default under DRIVE_ROOT)
    hf_home = os.getenv("HF_HOME", str(Path(drive_root) / ".hf"))
    transformers_cache = os.getenv("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))
    hf_datasets_cache = os.getenv("HF_DATASETS_CACHE", str(Path(hf_home) / "datasets"))
    torch_home = os.getenv("TORCH_HOME", str(Path(drive_root) / ".torch"))

    # Export cache envs so downstream libs pick them up
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache
    os.environ["TORCH_HOME"] = torch_home
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # Make sure everything exists
    for p in [
        Path(drive_root), Path(checkpoints), Path(data), Path(reports), Path(logs),
        Path(hf_home), Path(transformers_cache), Path(hf_datasets_cache), Path(torch_home)
    ]:
        p.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": drive_root,
        "checkpoints": checkpoints,
        "data": data,
        "reports": reports,
        "logs": logs,
        "hf_home": hf_home,
        "transformers_cache": transformers_cache,
        "hf_datasets_cache": hf_datasets_cache,
        "torch_home": torch_home,
        "mount_point": mount_point,
    }

    if verbose:
        print("📁 Storage initialized from .env:")
        for k in ["root","checkpoints","data","reports","logs","hf_home","transformers_cache","hf_datasets_cache","torch_home"]:
            print(f"  {k:>20}: {paths[k]}")

    return paths

if __name__ == "__main__":
    prepare_storage()

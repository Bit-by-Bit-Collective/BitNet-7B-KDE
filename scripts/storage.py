# scripts/storage.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from dotenv import load_dotenv

load_dotenv()


# ---------------------------
# Small helpers
# ---------------------------
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


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    return "" if v is None else v


# ---------------------------
# Remote URI builders
# ---------------------------
def _build_uri_s3(subpath: str) -> str:
    bucket = _env("S3_BUCKET", "")
    prefix = _env("S3_PREFIX", "")
    prefix = prefix.strip("/")
    parts = [p for p in (prefix, subpath.strip("/")) if p]
    tail = "/".join(parts)
    if not bucket:
        return f"s3://<missing-bucket>/{tail}"
    return f"s3://{bucket}/{tail}"


def _build_uri_webdav(base_url_env: str, root_path_env: str, subpath: str) -> str:
    base = _env(base_url_env, "").rstrip("/")
    root = _env(root_path_env, "/").strip("/")
    sub = subpath.strip("/")
    if not base:
        return f"webdav://<missing-base>/{root}/{sub}".rstrip("/")
    path = "/".join([p for p in (root, sub) if p])
    return f"{base}/{path}".rstrip("/")


def _build_uri_dropbox(subpath: str) -> str:
    root = _env("DROPBOX_ROOT", "/bitnet_poc").rstrip("/")
    sub = subpath if subpath.startswith("/") else f"/{subpath}"
    return f"dropbox://{root}{sub}"


def _build_uri_box(subpath: str) -> str:
    root_id = _env("BOX_ROOT_FOLDER_ID", "0")
    # Box uses folder IDs, so we just annotate
    return f"box://folder/{root_id}/{subpath.strip('/')}"


def _build_uri_onedrive(subpath: str) -> str:
    root = _env("ONEDRIVE_ROOT", "/BitNet-7B-KDE").strip("/")
    return f"onedrive://{root}/{subpath.strip('/')}"


def _build_uri_supabase(subpath: str) -> str:
    bucket = _env("SUPABASE_BUCKET", "bitnet-poc")
    prefix = _env("SUPABASE_PREFIX", "").strip("/")
    parts = [p for p in (prefix, subpath.strip("/")) if p]
    tail = "/".join(parts)
    return f"supabase://{bucket}/{tail}"


def _build_uri_firebase(subpath: str) -> str:
    bucket = _env("FIREBASE_STORAGE_BUCKET", "")
    if not bucket:
        return f"firebase://<missing-bucket>/{subpath.strip('/')}"
    return f"gs://{bucket}/{subpath.strip('/')}"


def _build_uri_mongodb(subpath: str) -> str:
    db = _env("MONGODB_DB_NAME", "bitnet_poc")
    coll = _env("MONGODB_COLLECTION_PREFIX", "artifacts_") + subpath.replace("/", "_")
    return f"mongodb+srv://{db}/{coll}"


def _build_uri_dynamodb(subpath: str) -> str:
    table = _env("DYNAMO_TABLE_PREFIX", "bitnet_poc_") + subpath.replace("/", "_")
    region = _env("DYNAMO_REGION", "")
    return f"dynamodb://{region}/{table}"


def _build_uri_gdrive(subpath: str) -> str:
    root = _required("DRIVE_ROOT").rstrip("/")
    return f"{root}/{subpath.strip('/')}"


def _build_uri_local(subpath: str) -> str:
    root = _required("DRIVE_ROOT").rstrip("/")
    return f"{root}/{subpath.strip('/')}"


# ---------------------------
# Main entry
# ---------------------------
def prepare_storage(verbose: bool = True) -> Dict[str, Any]:
    """
    Initializes local artifact directories and cache envs using ONLY .env values.
    If STORAGE_BACKEND=gdrive and running in Colab, mounts Drive when applicable.
    Returns:
      {
        'backend': <str>,
        'root': <local root>,
        'checkpoints'|'data'|'reports'|'logs': <local dirs>,
        'hf_home'|'transformers_cache'|'hf_datasets_cache'|'torch_home': <cache dirs>,
        'mount_point': <gdrive mount point (if any)>,
        'remote_root_uri': <string>,
        'build_remote_uri': <callable(subpath) -> str>
      }
    """
    backend = _env("STORAGE_BACKEND", "gdrive").strip().lower()
    # We still require a local root for staging/artifacts even for remote backends.
    drive_root = _required("DRIVE_ROOT").rstrip("/")

    # Auto-mount GDrive only if explicitly requested and path lives under mount
    mount_point = _env("GDRIVE_MOUNT_POINT", "/content/drive")
    auto_mount = _bool_env("AUTO_MOUNT_GDRIVE", True)
    should_mount = (
        backend == "gdrive"
        and _in_colab()
        and auto_mount
        and drive_root.startswith(mount_point.rstrip("/") + "/")
    )
    if should_mount:
        base = Path(mount_point)
        try:
            empty = (not base.exists()) or (base.exists() and not any(base.iterdir()))
        except Exception:
            empty = True
        if empty:
            if verbose:
                print(f"🔧 Mounting Google Drive at {mount_point}...")
            _mount_gdrive(mount_point)

    # Resolve core folders (from .env or fallback under DRIVE_ROOT)
    checkpoints = _env("CHECKPOINTS_DIR", f"{drive_root}/checkpoints")
    data = _env("DATA_DIR", f"{drive_root}/data")
    reports = _env("REPORTS_DIR", f"{drive_root}/reports")
    logs = _env("LOGS_DIR", f"{drive_root}/logs")

    # Caches (explicit from .env or default under DRIVE_ROOT)
    hf_home = _env("HF_HOME", f"{drive_root}/.hf")
    transformers_cache = _env("TRANSFORMERS_CACHE", f"{hf_home}/transformers")
    hf_datasets_cache = _env("HF_DATASETS_CACHE", f"{hf_home}/datasets")
    torch_home = _env("TORCH_HOME", f"{drive_root}/.torch")

    # Export cache envs so downstream libs pick them up
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache
    os.environ["TORCH_HOME"] = torch_home
    os.environ["TOKENIZERS_PARALLELISM"] = _env("TOKENIZERS_PARALLELISM", "false")
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _env("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    # Create local dirs
    auto_create = _bool_env("AUTO_CREATE_DIRS", True)
    if auto_create:
        for p in [
            Path(drive_root), Path(checkpoints), Path(data), Path(reports), Path(logs),
            Path(hf_home), Path(transformers_cache), Path(hf_datasets_cache), Path(torch_home)
        ]:
            p.mkdir(parents=True, exist_ok=True)

    # Pick remote URI builder based on backend
    builder: Callable[[str], str]
    remote_root_uri: str
    if backend == "gdrive":
        builder = _build_uri_gdrive
        remote_root_uri = builder("")  # equals DRIVE_ROOT
    elif backend == "local":
        builder = _build_uri_local
        remote_root_uri = builder("")
    elif backend == "s3":
        builder = _build_uri_s3
        # compose a root-ish URI for display
        prefix = _env("S3_PREFIX", "").strip("/")
        bucket = _env("S3_BUCKET", "")
        remote_root_uri = f"s3://{bucket}/{prefix}".rstrip("/")
    elif backend == "nextcloud":
        builder = lambda sp: _build_uri_webdav("NC_WEBDAV_URL", "NC_ROOT_PATH", sp)
        remote_root_uri = builder("")
    elif backend == "webdav":
        builder = lambda sp: _build_uri_webdav("WEBDAV_URL", "WEBDAV_ROOT_PATH", sp)
        remote_root_uri = builder("")
    elif backend == "dropbox":
        builder = _build_uri_dropbox
        remote_root_uri = builder("")  # dropbox://<root>
    elif backend == "box":
        builder = _build_uri_box
        remote_root_uri = builder("")  # box://folder/<id>
    elif backend == "onedrive":
        builder = _build_uri_onedrive
        remote_root_uri = builder("")
    elif backend == "supabase":
        builder = _build_uri_supabase
        remote_root_uri = builder("")
    elif backend == "firebase":
        builder = _build_uri_firebase
        remote_root_uri = builder("")
    elif backend == "mongodb_atlas":
        builder = _build_uri_mongodb
        remote_root_uri = builder("root")
    elif backend == "dynamodb":
        builder = _build_uri_dynamodb
        remote_root_uri = builder("root")
    elif backend == "icloud":
        # Placeholder (no official WebDAV). Treat like an annotated remote.
        builder = lambda sp: f"icloud://{sp.strip('/')}"
        remote_root_uri = builder("")
    else:
        raise RuntimeError(f"Unsupported STORAGE_BACKEND: {backend}")

    # Build return payload
    paths: Dict[str, Any] = {
        "backend": backend,
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
        "remote_root_uri": remote_root_uri,
        "build_remote_uri": builder,  # callable(subpath) -> str
    }

    if verbose:
        print("📁 Storage initialized from .env")
        print(f"  backend           : {backend}")
        print(f"  local root        : {paths['root']}")
        print(f"  checkpoints       : {paths['checkpoints']}")
        print(f"  data              : {paths['data']}")
        print(f"  reports           : {paths['reports']}")
        print(f"  logs              : {paths['logs']}")
        print(f"  hf_home           : {paths['hf_home']}")
        print(f"  transformers_cache: {paths['transformers_cache']}")
        print(f"  hf_datasets_cache : {paths['hf_datasets_cache']}")
        print(f"  torch_home        : {paths['torch_home']}")
        print(f"  remote_root_uri   : {paths['remote_root_uri']}")

        # Heads-up for remote backends that need SDKs (we only build URIs here).
        if backend in {"s3", "dropbox", "box", "nextcloud", "webdav", "supabase", "firebase"}:
            print("ℹ️  Note: remote backends are URI-only here. Sync/upload is handled by calling code.")
            print("    Make sure the relevant SDK/credentials are configured if you implement sync:")

            if backend == "s3":
                print("    - pip install boto3 ; uses AWS_* and S3_* from .env")
            elif backend in {"nextcloud", "webdav"}:
                print("    - pip install requests ; WEBDAV/NC_* creds from .env")
            elif backend == "dropbox":
                print("    - pip install dropbox ; uses DROPBOX_ACCESS_TOKEN")
            elif backend == "box":
                print("    - pip install boxsdk ; uses BOX_DEVELOPER_TOKEN (dev) or JWT app")
            elif backend == "supabase":
                print("    - pip install supabase ; uses SUPABASE_URL/keys")
            elif backend == "firebase":
                print("    - pip install google-cloud-storage ; uses FIREBASE_*")

    return paths


if __name__ == "__main__":
    prepare_storage()

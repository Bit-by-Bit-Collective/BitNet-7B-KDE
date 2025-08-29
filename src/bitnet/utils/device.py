# src/bitnet/utils/device.py
"""Enhanced device selection with support for CUDA, ROCm, MPS, and CPU."""

import os
import torch
from typing import Tuple, Optional


def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """
    Smart device and dtype selection based on environment and hardware.
    
    Returns:
        (device, dtype) tuple
    """
    device_type = os.getenv("DEVICE_TYPE", "auto").lower()
    force_cpu = os.getenv("FORCE_CPU", "0") == "1"
    
    # Dtype selection
    dtype_str = os.getenv("TORCH_DTYPE", "bf16").lower()
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    
    # Force CPU if requested
    if force_cpu:
        print("🔧 Forced CPU mode (FORCE_CPU=1)")
        return torch.device("cpu"), torch.float32
    
    # Auto device selection
    if device_type == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Check if bf16 is supported
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                print("⚠️ BF16 not supported on this GPU, falling back to FP16")
                dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            # MPS doesn't support bf16 yet
            if dtype == torch.bfloat16:
                print("⚠️ BF16 not supported on MPS, falling back to FP32")
                dtype = torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float32
    
    # Explicit device selection
    elif device_type == "cuda":
        if not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
            dtype = torch.float32
        else:
            device = torch.device("cuda")
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                dtype = torch.float16
    
    elif device_type == "rocm":
        # AMD ROCm support (PyTorch built with ROCm)
        if torch.cuda.is_available():  # ROCm uses CUDA API
            device = torch.device("cuda")
            print("🔧 Using ROCm/AMD GPU")
        else:
            print("⚠️ ROCm requested but not available, falling back to CPU")
            device = torch.device("cpu")
            dtype = torch.float32
    
    elif device_type == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            if dtype == torch.bfloat16:
                dtype = torch.float32
        else:
            print("⚠️ MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
            dtype = torch.float32
    
    elif device_type == "cpu":
        device = torch.device("cpu")
        dtype = torch.float32
    
    else:
        print(f"⚠️ Unknown device type: {device_type}, using CPU")
        device = torch.device("cpu")
        dtype = torch.float32
    
    # Print device info
    if device.type == "cuda":
        print(f"🖥️ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_capability()}")
    elif device.type == "mps":
        print("🖥️ Using Apple Silicon GPU (MPS)")
    else:
        print("🖥️ Using CPU")
    
    print(f"📊 Dtype: {dtype}")
    
    return device, dtype


def get_mixed_precision_settings(device: torch.device, dtype: torch.dtype) -> dict:
    """
    Get appropriate mixed precision settings for the device/dtype combo.
    
    Returns:
        Dictionary with AMP settings
    """
    use_amp = os.getenv("USE_AMP", "1") == "1"
    
    if not use_amp or device.type == "cpu":
        return {"enabled": False, "dtype": None, "backend": None}
    
    if device.type == "cuda":
        if dtype == torch.float16:
            return {
                "enabled": True,
                "dtype": torch.float16,
                "backend": "native",
                "use_grad_scaler": True
            }
        elif dtype == torch.bfloat16:
            return {
                "enabled": True,
                "dtype": torch.bfloat16,
                "backend": "native",
                "use_grad_scaler": False  # GradScaler not needed for bf16
            }
    elif device.type == "mps":
        # MPS has limited AMP support
        return {
            "enabled": False,  # Often better to not use AMP on MPS
            "dtype": None,
            "backend": None
        }
    
    return {"enabled": False, "dtype": None, "backend": None}


def setup_device_env():
    """Set up environment variables for optimal device usage."""
    
    # CUDA optimizations
    if torch.cuda.is_available():
        # Enable TF32 for Ampere GPUs (3090, A100, etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cudNN autotuner for better performance
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction if specified
        mem_frac = os.getenv("CUDA_MEMORY_FRACTION")
        if mem_frac:
            torch.cuda.set_per_process_memory_fraction(float(mem_frac))
    
    # MPS optimizations
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Enable fallback for unsupported ops
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

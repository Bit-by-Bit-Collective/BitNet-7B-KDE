#!/usr/bin/env python3
"""Quick POC test for BitNet-7B-KDE with AIMLAPI."""

import os
import sys
import json
from pathlib import Path

def test_aimlapi():
    """Test AIMLAPI connection and logprobs."""
    try:
        import openai
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or not api_key.startswith("aimlapi_"):
            print("❌ AIMLAPI key not found or invalid")
            print("   Set OPENAI_API_KEY=aimlapi_YOUR_KEY in .env")
            return False
        
        print("🔑 AIMLAPI key found")
        
        # Test API
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.aimlapi.com/v1"
        )
        
        print("📡 Testing AIMLAPI connection...")
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-1B-Instruct-Turbo",  # Small model for testing
            messages=[{"role": "user", "content": "Say 'test successful' in 3 words"}],
            max_tokens=10,
            temperature=0,
            logprobs=True,
            top_logprobs=5
        )
        
        content = response.choices[0].message.content
        has_logprobs = response.choices[0].logprobs is not None
        
        print(f"✅ Response: {content}")
        print(f"✅ Logprobs available: {has_logprobs}")
        
        if has_logprobs and response.choices[0].logprobs.content:
            first_token = response.choices[0].logprobs.content[0]
            print(f"   First token: '{first_token.token}' (logprob: {first_token.logprob:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ AIMLAPI test failed: {e}")
        return False

def create_poc_structure():
    """Create necessary directories and files for POC."""
    
    # Create directories
    dirs = [
        "data/teacher",
        "data/kd", 
        "data/eval",
        "checkpoints",
        "logs",
        "outputs"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created {len(dirs)} directories")
    
    # Generate sample prompts
    prompts = [
        {"prompt": "What is machine learning?"},
        {"prompt": "Write a Python hello world"},
        {"prompt": "Explain gravity simply"},
        {"prompt": "What is 2+2?"},
        {"prompt": "Create a JSON user object"},
    ]
    
    with open("data/prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)
    
    print(f"✅ Created {len(prompts)} test prompts")
    
    return True

def check_environment():
    """Check if environment is properly configured."""
    
    print("\n🔍 Environment Check")
    print("=" * 40)
    
    # Check Python version
    py_version = sys.version.split()[0]
    print(f"Python: {py_version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check key dependencies
    deps = ["transformers", "openai", "pyarrow", "datasets"]
    missing = []
    
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep} installed")
        except ImportError:
            print(f"❌ {dep} missing")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠️ Install missing: pip install {' '.join(missing)}")
        return False
    
    return True

def run_mini_pipeline():
    """Run a mini version of the full pipeline for testing."""
    
    print("\n🚀 Running Mini Pipeline")
    print("=" * 40)
    
    # Step 1: Generate prompts
    print("\n1️⃣ Generating prompts...")
    if not Path("data/prompts.json").exists():
        create_poc_structure()
    
    # Step 2: Test teacher
    print("\n2️⃣ Testing teacher model...")
    if not test_aimlapi():
        return False
    
    # Step 3: Check if we can import our modules
    print("\n3️⃣ Checking module imports...")
    try:
        from src.bitnet.models import BitNetLM
        from src.bitnet.data import KDTraceDataset
        from src.bitnet.losses import combined_loss
        print("✅ All modules importable")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Step 4: Test model creation
    print("\n4️⃣ Testing model creation...")
    try:
        import torch
        from src.bitnet.models import BitNetLM
        
        model = BitNetLM(
            vocab_size=32000,
            dim=256,  # Very small for testing
            n_layers=2,
            n_heads=4,
            head_dim=64
        )
        
        # Test forward pass
        dummy_input = torch.randint(0, 32000, (1, 10))
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
        print(f"✅ Forward pass successful: output shape {output.shape}")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    return True

def main():
    """Main POC test runner."""
    
    print("🧪 BitNet-7B-KDE POC Test")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed")
        return 1
    
    # Create structure
    create_poc_structure()
    
    # Run mini pipeline
    if not run_mini_pipeline():
        print("\n❌ Mini pipeline failed")
        return 1
    
    print("\n" + "=" * 40)
    print("✅ POC test successful!")
    print("\nNext steps:")
    print("1. Run: make teacher    # Collect teacher samples")
    print("2. Run: make collect    # Collect KD traces")
    print("3. Run: make train      # Train mini model")
    print("4. Run: make eval       # Evaluate model")
    print("\nOr run: make pipeline   # Run everything")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

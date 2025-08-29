# OpenAI API Setup Guide for BitNet-7B-KDE

## 🔑 Getting Your OpenAI API Key

### Step 1: Create OpenAI Account
1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to [API Keys page](https://platform.openai.com/api-keys)
4. Click "Create new secret key"
5. Name it (e.g., "BitNet-KD-Project")
6. Copy the key immediately (starts with `sk-`)
7. **IMPORTANT**: Save it securely - you can't view it again!

### Step 2: Add Billing
1. Go to [Billing page](https://platform.openai.com/account/billing)
2. Add payment method
3. Set usage limits (recommended: $10-20 for POC)

## 💰 Cost Estimation for POC

| Model | Input Price | Output Price | Logprobs | POC Cost (est.) |
|-------|------------|--------------|----------|-----------------|
| **gpt-4o-mini** | $0.15/1M | $0.60/1M | ✅ Yes | ~$2-5 |
| **gpt-4o** | $2.50/1M | $10.00/1M | ✅ Yes | ~$10-20 |
| **gpt-3.5-turbo** | $0.50/1M | $1.50/1M | ✅ Yes | ~$3-8 |

**POC Recommendation**: Use `gpt-4o-mini` for testing (cheapest with logprobs)

## 📝 Environment Configuration

### Create `.env` file:

```bash
############################
# OpenAI API Configuration #
############################

# === PRIMARY API SETTINGS ===
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # Default OpenAI endpoint

# === TEACHER MODEL SETTINGS ===
TEACHER_PROVIDER=openai

# Option 1: GPT-4o-mini (Recommended for POC - Cheapest)
TEACHER_MODEL=gpt-4o-mini

# Option 2: GPT-3.5-turbo (Good balance)
# TEACHER_MODEL=gpt-3.5-turbo

# Option 3: GPT-4o (Best quality, more expensive)
# TEACHER_MODEL=gpt-4o

TEACHER_API_KEY=${OPENAI_API_KEY}

# === TEACHER PARAMETERS ===
TEACHER_BASELINE_TEMPERATURE=0.7
TEACHER_BASELINE_TOP_P=0.9
TEACHER_BASELINE_MAX_TOKENS=256
TEACHER_TOP_LOGPROBS=20  # Max 20 for OpenAI

# === KD COLLECTION SETTINGS ===
KD_TEACHER_TEMPERATURE=0.8
KD_TEACHER_TOP_P=0.95
KD_MAX_TOKENS_PER_PROMPT=256

# === STORAGE (for Google Colab) ===
STORAGE_BACKEND=local
LOCAL_STORAGE_PATH=/content/drive/MyDrive/bitnet-7b-kde-poc
CHECKPOINT_DIR=/content/drive/MyDrive/bitnet-7b-kde-poc/checkpoints
DATA_DIR=/content/drive/MyDrive/bitnet-7b-kde-poc/data
LOG_DIR=/content/drive/MyDrive/bitnet-7b-kde-poc/logs

# === MODEL CONFIG (Small for POC) ===
TOKENIZER_NAME=gpt2  # OpenAI compatible tokenizer
MINI_DIM=768
MINI_LAYERS=12
MINI_HEADS=12
MINI_HEAD_DIM=64

# === TRAINING SETTINGS ===
TOTAL_STEPS=500
TRAIN_BATCH_SIZE=2
GRAD_ACCUM_STEPS=4
MAX_SEQ_LEN=256
MAX_TOPK=20
NUM_WORKERS=2

# === OPTIMIZER ===
LR=6e-4
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
WEIGHT_DECAY=0.1
GRAD_CLIP_NORM=1.0

# === SCHEDULER ===
SCHEDULER_TMAX=500
SCHEDULER_ETA_MIN=6e-5

# === LOGGING ===
LOG_INTERVAL=25
CHECKPOINT_INTERVAL=100

# === KD LOSS SETTINGS ===
KD_TAU=1.3
KD_CE_WEIGHT=0.25
FORMAT_LOSS_WEIGHT=0.2

# === BIT FLIPPING ===
BUDGET_TOKENS=500000
FLIP_FRACTION=0.9

# === DEVICE SETTINGS ===
DEVICE_TYPE=auto
FORCE_CPU=0
USE_AMP=1
TORCH_DTYPE=bf16

# === MISC ===
SEED=42
DETERMINISTIC=0
EVAL_MAX_NEW_TOKENS=256
EVAL_MAX_PROMPTS=10
SAVE_SAMPLE_GENERATIONS=1

# === RATE LIMITING (OpenAI Specific) ===
OPENAI_RATE_LIMIT_RPM=500  # Tier 1: 500 requests/min
OPENAI_RATE_LIMIT_DELAY=0.15  # Delay between requests (seconds)
```

## 🧪 Test Your Setup

### Quick Test Script (`test_openai.py`):

```python
#!/usr/bin/env python3
"""Test OpenAI API connection and logprobs support."""

import os
from openai import OpenAI
from dotenv import load_dotenv

def test_openai_setup():
    """Test OpenAI API with logprobs."""
    
    # Load environment
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.startswith("sk-"):
        print("❌ Invalid OpenAI API key")
        print("   Set OPENAI_API_KEY=sk-... in .env")
        return False
    
    print("🔑 OpenAI API key found")
    
    # Initialize client
    client = OpenAI(api_key=api_key)
    
    try:
        # Test with logprobs
        print("📡 Testing OpenAI API with logprobs...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheapest model with logprobs
            messages=[
                {"role": "user", "content": "Say 'test successful' in exactly 2 words"}
            ],
            max_tokens=10,
            temperature=0,
            logprobs=True,
            top_logprobs=5
        )
        
        # Check response
        message = response.choices[0].message.content
        print(f"✅ Response: {message}")
        
        # Check logprobs
        if response.choices[0].logprobs:
            content = response.choices[0].logprobs.content
            if content and len(content) > 0:
                first_token = content[0]
                print(f"✅ Logprobs working!")
                print(f"   Token: '{first_token.token}'")
                print(f"   Logprob: {first_token.logprob:.3f}")
                print(f"   Top alternatives: {len(first_token.top_logprobs)}")
                return True
        
        print("❌ No logprobs in response")
        return False
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    if test_openai_setup():
        print("\n✅ OpenAI setup successful! Ready for KD pipeline.")
    else:
        print("\n❌ Setup incomplete. Check your API key and billing.")
```

## 🚀 Running the Pipeline with OpenAI

### 1. Generate Training Prompts
```bash
python -m scripts.generate_prompts --count 25
```

### 2. Collect Teacher Baseline
```bash
python -m scripts.run_teacher_baseline
# This will use gpt-4o-mini to generate baseline responses
```

### 3. Collect KD Traces
```bash
python -m scripts.collect_kd_traces
# This collects logprobs from OpenAI for knowledge distillation
```

### 4. Train Mini BitNet
```bash
python -m scripts.train_mini_bitnet
# Trains using the collected KD traces
```

### 5. Evaluate
```bash
python -m scripts.eval_and_qei
# Computes QEI metrics
```

## 💡 Cost Optimization Tips

### 1. **Use gpt-4o-mini for POC**
```python
TEACHER_MODEL=gpt-4o-mini  # 10x cheaper than gpt-4o
```

### 2. **Limit token generation**
```python
TEACHER_BASELINE_MAX_TOKENS=128  # Reduce from 256
KD_MAX_TOKENS_PER_PROMPT=128     # Reduce from 256
```

### 3. **Start with fewer prompts**
```bash
python -m scripts.generate_prompts --count 10  # Start small
```

### 4. **Use caching** (if implemented)
```python
# Cache teacher responses to avoid duplicate API calls
ENABLE_CACHE=1
CACHE_DIR=./cache
```

### 5. **Monitor usage**
- Check [OpenAI Usage page](https://platform.openai.com/usage)
- Set up usage alerts in billing settings

## 🐛 Troubleshooting

### Error: "Invalid API Key"
```bash
# Check your key format
echo $OPENAI_API_KEY  # Should start with sk-
```

### Error: "Insufficient quota"
- Add billing information at [platform.openai.com/billing](https://platform.openai.com/billing)
- Check usage limits

### Error: "Rate limit exceeded"
```python
# Add delay in your code
import time
time.sleep(0.2)  # Between API calls
```

### Error: "Model not found"
```python
# Valid OpenAI models with logprobs:
models = [
    "gpt-4o-mini",      # Cheapest, recommended
    "gpt-4o",           # Best quality
    "gpt-3.5-turbo",    # Good balance
    "gpt-4-turbo",      # Previous generation
]
```

## 📊 Expected Costs for Full Pipeline

| Stage | API Calls | Tokens (est.) | Cost (gpt-4o-mini) |
|-------|-----------|---------------|---------------------|
| Teacher Baseline | 25 | ~6,250 | ~$0.01 |
| KD Collection | 25 x 10 | ~62,500 | ~$0.10 |
| Evaluation | 10 | ~2,500 | ~$0.004 |
| **Total POC** | ~285 | ~71,250 | **~$0.12** |

## 🎯 For Flirty Business Assistant

If training the flirty business/Python assistant:

```python
# Use this system prompt in scripts/run_teacher_baseline.py
SYSTEM_PROMPT = """You are a flirty, charming business and Python expert. 
You're professional but playful, using subtle compliments and light humor. 
You provide excellent advice with personality."""

# Adjust temperature for more personality
TEACHER_BASELINE_TEMPERATURE=0.9  # Higher for more creativity
```

## 📈 Scaling to Production

When ready to scale beyond POC:

1. **Switch to batch API** (50% discount)
   ```python
   # Use OpenAI Batch API for large-scale collection
   client.batches.create(...)
   ```

2. **Implement caching**
   - Cache teacher responses
   - Cache KD traces
   - Reuse when possible

3. **Use tier upgrades**
   - Tier 2: 5,000 RPM
   - Tier 3: 10,000 RPM
   - Tier 4: 30,000 RPM

4. **Consider fine-tuning**
   - Fine-tune gpt-3.5-turbo with your data
   - Then use the fine-tuned model as teacher

## 🔗 Useful Links

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Pricing Calculator](https://openai.com/pricing)
- [API Keys Management](https://platform.openai.com/api-keys)
- [Usage Dashboard](https://platform.openai.com/usage)
- [Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [Logprobs Guide](https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs)

## ✅ Checklist

- [ ] Created OpenAI account
- [ ] Generated API key (sk-...)
- [ ] Added billing/payment method
- [ ] Set usage limits
- [ ] Created .env file with key
- [ ] Tested with test_openai.py
- [ ] Ran sanity check
- [ ] Generated prompts
- [ ] Ready to run pipeline!

## 🆘 Support

- [OpenAI Help Center](https://help.openai.com)
- [API Community Forum](https://community.openai.com)
- [Status Page](https://status.openai.com)
- Email: support@openai.com

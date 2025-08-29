# AIMLAPI Setup Guide for BitNet-7B-KDE

## Getting Your AIMLAPI Key

1. **Sign up at [AIMLAPI](https://aimlapi.com/)**
2. Navigate to your dashboard
3. Generate an API key
4. Copy the key (starts with `aimlapi_`)

## Available Models on AIMLAPI

AIMLAPI provides access to multiple models that support logprobs:

### Recommended for Teacher (Good logprobs support):
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` (Best quality)
- `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` (Faster, cheaper)
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.2`

### For Testing (Smaller/Faster):
- `meta-llama/Llama-3.2-3B-Instruct-Turbo`
- `meta-llama/Llama-3.2-1B-Instruct-Turbo`

## Configuration in .env

```bash
# AIMLAPI Configuration
OPENAI_API_KEY=aimlapi_YOUR_KEY_HERE
OPENAI_BASE_URL=https://api.aimlapi.com/v1

# Teacher model (using AIMLAPI)
TEACHER_PROVIDER=openai
TEACHER_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
TEACHER_API_KEY=aimlapi_YOUR_KEY_HERE
```

## Testing Your Setup

```python
# Quick test script
import openai

client = openai.OpenAI(
    api_key="aimlapi_YOUR_KEY_HERE",
    base_url="https://api.aimlapi.com/v1"
)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Say hello"}],
    max_tokens=10,
    logprobs=True,
    top_logprobs=5
)

print(response.choices[0].message.content)
print("Logprobs available:", response.choices[0].logprobs is not None)
```

## Cost Optimization Tips

1. **Start with smaller models** for testing:
   - Use `Llama-3.2-3B` instead of `Llama-3.1-70B` initially
   
2. **Limit tokens** during POC:
   - Set `TEACHER_BASELINE_MAX_TOKENS=128` for testing
   - Set `KD_MAX_TOKENS_PER_PROMPT=128`

3. **Use fewer prompts**:
   - Start with 10-20 prompts for initial testing
   - Scale up once pipeline is verified

## Troubleshooting

### Error: "Invalid API Key"
- Ensure key starts with `aimlapi_`
- Check for trailing spaces
- Verify key in AIMLAPI dashboard

### Error: "Model not found"
- Use exact model names from list above
- Check AIMLAPI docs for latest models

### Error: "No logprobs in response"
- Ensure `logprobs=True` in request
- Set `top_logprobs=20` (max value)
- Some models may not support logprobs

### Rate Limits
- AIMLAPI has generous rate limits
- If hit, add delays between requests:
  ```python
  import time
  time.sleep(0.5)  # Between API calls
  ```

## Google Colab Secrets Setup

1. In Colab, click the 🔑 key icon in left sidebar
2. Add a new secret named `AIMLAPI_KEY`
3. Paste your API key as the value
4. Access in code:
   ```python
   from google.colab import userdata
   api_key = userdata.get('AIMLAPI_KEY')
   ```

## Monitoring Usage

Check your AIMLAPI dashboard for:
- Token usage
- API call count
- Remaining credits
- Cost breakdown by model

## Support

- AIMLAPI Discord: [Join here](https://discord.gg/aimlapi)
- Documentation: [docs.aimlapi.com](https://docs.aimlapi.com)
- Email: support@aimlapi.com

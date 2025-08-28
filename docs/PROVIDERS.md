# Providers & Capabilities

| Provider   | Base URL (default)                | logprobs | top_logprobs | notes                  |
|------------|-----------------------------------|----------|--------------|------------------------|
| OpenAI     | api.openai.com                    | ✅        | ✅            | best tested            |
| Anthropic  | api.anthropic.com                 | ⚠️*       | ⚠️*           | limited logprobs; TBD  |
| Groq       | api.groq.com/openai/v1            | ✅        | ✅            | OpenAI-compatible      |
| AIMLAPI    | api.aimlapi.com/v1                | ✅        | ✅            | OpenAI-compatible      |
| Gemini     | generativelanguage.googleapis.com | ⚠️*       | ⚠️*           | evolving logprobs API  |

> Configure with `.env`: `*_API_KEY`, `*_BASE_URL`, `PROVIDER` and `TEACHER_*`.
> If provider lacks logprobs/top-k, skip KD collection with that provider.

# Environment Variable Template for YouTube Pipeline

This template describes the expected layout and required variables for your `.env` file.  
Copy this content to a file named `.env` in your project root and fill in your actual keys/tokens.

---

## Required

```
# OpenAI API key for Whisper transcription (required for audio-to-text)
OPENAI_API_KEY="sk-..."

# OpenRouter API key for LLM correction/summarization (required for all LLM tasks)
OPENROUTER_API_KEY="sk-or-..."
```

## Optional

```
# Hugging Face Hub token (required for diarization with gated models like pyannote/speaker-diarization-3.1)
HF_TOKEN="hf_..."
```

## Notes

- Both `OPENAI_API_KEY` and `OPENROUTER_API_KEY` are required for the full pipeline to function.
- `HF_TOKEN` is only required if you want to use speaker diarization and have not logged in via `huggingface-cli login`.
- Never commit your `.env` file with real keys to version control.

---

**Example .env file:**

```
OPENAI_API_KEY="ssk-proj-NdfPUff3PKzghMnxy2_5W1iZ0-9qD1JlxRtRtKe7GTX_vFQ-vW-hXaLpA7t0PYa3n-QRlNnBz2T3BlbkFJjY5C4BrartcPDGbwdRzUap0DxzzTSWPNg1UYIF-TbGY5Npd02YylbH74uaYq4LoZItml_3-ysA"
OPENROUTER_API_KEY="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"
HF_TOKEN="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"

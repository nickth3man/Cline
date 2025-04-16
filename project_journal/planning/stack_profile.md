# Detected Stack Profile

## Languages
- 🐍 **Python 3.8+** (primary language for all workflow, orchestration, and scripting)

## Frameworks/Libraries
- **yt-dlp**: YouTube video/audio download
- **OpenAI API**: Whisper (`whisper-1`) for transcription (official endpoint, not via OpenRouter)
- **OpenRouter API**: LLM-based correction and summarization (user-selectable, live-fetched, advanced selection)
- **pyannote.audio**: Local speaker diarization (requires Hugging Face token)
- **SpaCy**: Sentence splitting and text cleanup
- **PyQt6** (or web-based UI): User interface for advanced model selection, live price display, error feedback
- **python-dotenv**: Environment variable management
- **Standard Python libraries**: os, subprocess, logging, re, datetime, json, csv, etc.
- **Testing**: pytest, pytest-mock

## Build Tools & Automation
- **Batch scripts**: run_pipeline.bat, verify_structure.bat for environment and pipeline orchestration
- **Python wrapper scripts**: For consistent interpreter usage and import management

## CI/CD
- No explicit CI/CD configuration detected in the current context (recommendation: add GitHub Actions or similar for automated testing and deployment)

## Databases/ORMs
- No explicit database/ORM detected; optional use of SQLite for centralized index/metadata (per systemPatterns.md)

## Configuration & Environment
- **YAML**: config/config.yaml for model and API configuration
- **.env**: Environment variable management

## Not Detected / Not Used
- No evidence of: React, Vue, Angular, Django, Flask, Laravel, Spring, Express, Next, Nuxt, SvelteKit, Tailwind, Bootstrap, or other web frameworks in project source

## Potential Specialist Modes Needed
- 🐍 Python Specialist
- 🧠 LLM/AI Integration Specialist (OpenAI, OpenRouter)
- 🗣️ Audio Processing Specialist (yt-dlp, pyannote.audio, Whisper)
- 🧪 Test Automation Specialist (pytest)
- 🎨 UI/UX Specialist (PyQt6 or web-based UI)
- ⚙️ DevOps/Environment Specialist (virtualenv, dependency management, batch scripting)
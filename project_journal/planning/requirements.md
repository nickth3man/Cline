# Project Requirements

## Core Functionality

- Automate the process of downloading, transcribing, correcting, summarizing, and diarizing YouTube videos and playlists.
- For each video:
  - Download audio using yt-dlp and save in a per-video folder (named after the video title).
  - Transcribe audio using the official OpenAI Whisper API (`whisper-1`) with `OPENAI_API_KEY` (not via OpenRouter).
  - Perform sentence splitting and cleanup using SpaCy (required).
  - Allow the user to select any available OpenRouter LLM for correction and summarization, with live price calculation, search/filter, model variants, provider routing, and model details in the UI.
  - Perform diarization locally using `pyannote.audio` after transcription.
  - Enforce a strict output structure: each video folder ONLY contains (1) the original video file, (2) the transcript file with _corrected.md suffix, (3) the summary file with _summary.md suffix. No other files or folders are allowed.
  - Save all outputs (audio, raw transcript, corrected transcript, summary, metadata, diarization, HTML reader) in the video’s folder.
- Provide a user-friendly UI (PyQt6 or web-based) with live model selection, real-time cost feedback, advanced model selection, and robust error handling.
- All errors (API, network, cache, model selection) are surfaced in the GUI with clear troubleshooting steps.
- All assets for a video are grouped together for easy access and future use.
- The workflow is extensible, allowing for new features (e.g., additional metadata, export formats) to be added without disrupting the core organization.

## Design & Aesthetics

- All files for a video are grouped together, making navigation and management simple.
- The structure is robust to batch processing and future scaling.
- The process is transparent, with clear logging, error reporting, and per-video error logs for easy troubleshooting.
- UI must provide advanced model selection, real-time price feedback, and robust error handling.
- Strict alignment with OpenRouter's API and best practices for model selection, error handling, and user feedback.

## Technical Aspects

- Only `openai/whisper-large-v3` is used for transcription (audio-to-text).
- Only SpaCy is used for sentence splitting (no regex fallback).
- LLM correction and summarization models are user-selectable from all OpenRouter LLMs, fetched live, with advanced selection and error handling.
- pyannote.audio must be set up locally for diarization; requires a Hugging Face token and model downloads.
- OpenRouter API costs for transcription and LLM steps depend on audio length and model selection; UI must display real-time price estimates and surface all errors.
- Large playlists or long videos may require significant disk space, compute, and API usage.
- All scripts are run from the project root; outputs are organized per video.
- Documentation (memory bank) must be kept up to date to ensure continuity after resets and must always reflect the current OpenRouter-aligned implementation.
- The system should be easy to extend with new processing steps or integrations, and should provide clear error reporting and handling, surfaced in the GUI.

## Out of Scope

- Manual download or organization of files.
- Scattered outputs or inconsistent naming.
- Features not explicitly requested or documented in the memory bank.
- Skipping diarization or using cloud-based diarization APIs (unless added in future planning). User selection of diarization method is not supported.
- Any deviation from OpenRouter's API, model selection, or best practices.
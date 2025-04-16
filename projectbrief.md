# Project Brief

## Core Requirements and Goals

- Automate the process of downloading, transcribing, correcting, summarizing, and diarizing YouTube videos and playlists.
- For each video, download audio and save it in a per-video folder.
- Transcribe audio using the official OpenAI Whisper API (`whisper-1`) with `OPENAI_API_KEY` (not via OpenRouter, due to lack of endpoint).
- Perform sentence splitting and cleanup using SpaCy (required).
- Allow the user to select any available OpenRouter LLM for correction and summarization, with live price calculation, search/filter, model variants, provider routing, and model details in the UI.
- Perform diarization locally using `pyannote.audio` after transcription.
- Enforce a strict output structure where each video folder ONLY contains: (1) The original video file, (2) The transcript file (with _corrected.md suffix), and (3) The transcript summary file (with _summary.md suffix). No other files or folders are allowed.
- Provide a user-friendly UI with live model selection, real-time cost feedback, advanced model selection, and robust error handling.
- Ensure all assets for a video are grouped together for easy access and future use.
- Strictly align LLM selection, cost calculation, and interaction components with OpenRouter's API and best practices.

## Source of Truth for Project Scope

- The workflow must be fully automated and require minimal manual intervention.
- File and folder naming must be consistent, predictable, and robust to batch operations.
- The system should be easy to extend with new processing steps or integrations, and should provide clear error reporting and handling, surfaced in the GUI.
- Documentation (memory bank) must be kept up to date to ensure continuity after resets and must always reflect the current OpenRouter-aligned implementation.
- Model selection for LLM steps must be live, advanced, and transparent, with real-time cost feedback, search/filter, variants, provider routing, and model details.
- Diarization is always included and performed locally using `pyannote.audio`.

## Out of Scope

- Manual download or organization of files.
- Scattered outputs or inconsistent naming.
- Features not explicitly requested or documented in the memory bank.
- Skipping diarization or using cloud-based diarization APIs (unless added in future planning). User selection of diarization method is not supported.
- Any deviation from OpenRouter's API, model selection, or best practices.

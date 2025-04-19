# YouTube Video Pipeline

A comprehensive tool for downloading, transcribing, and summarizing YouTube videos with speaker diarization.

## Features

- Download videos from YouTube playlists
- Transcribe audio using WhisperX (self-hosted)
- Perform speaker diarization using pyannote.audio
- Correct and format transcripts using LLMs via OpenRouter
- Generate summaries of video content
- User-friendly GUI interface

## Project Structure (:LayeredArchitecture)

This project follows a **:LayeredArchitecture** pattern to promote separation of concerns, maintainability, and testability. The **:DirectoryStructure** reflects these layers:

```text
.
├── config/                 # Configuration files (API keys, model settings, pipeline parameters)
├── docs/                   # Project documentation (including this README, architecture details)
├── logs/                   # Application log files for debugging and monitoring
├── output/                 # Default directory for processed outputs (downloads, transcripts, summaries)
├── scripts/                # Helper scripts (e.g., setup, verification, batch processing utilities)
├── src/                    # Main source code, structured by layers:
│   ├── core/               # Core business logic: Pipeline orchestration, domain models, core algorithms (e.g., transcription logic). Independent of UI and external tools.
│   ├── adapters/           # Adapters/Interfaces to external services & libraries: Modules for interacting with YouTube APIs, WhisperX, LLMs (OpenRouter), file system, etc. Decouples core logic from specific tools.
│   ├── presentation/       # Presentation layer: User interface components (e.g., GUI implemented with Tkinter/PyQt, CLI entry points and argument parsing). Interacts with the core layer.
│   ├── common/             # Shared utilities, data structures (DTOs), constants, or helper functions used across multiple layers.
│   └── __init__.py         # Makes 'src' a Python package.
├── tests/                  # Automated tests (unit, integration, end-to-end). Structure may mirror 'src/'. Contains tests ensuring component correctness and system stability (part of our cumulative testing strategy).
├── .env.example            # Example environment file template. Copy to .env and fill in secrets.
├── .gitignore              # Specifies intentionally untracked files that Git should ignore.
├── requirements.txt        # Python package dependencies for reproducibility (:Technology choices).
├── run_pipeline.bat        # Convenience script for running the CLI pipeline on Windows.
└── README.md               # This file - overview, setup, and usage.
```

This structure helps isolate changes. For example, updating the GUI (`src/presentation`) should not require changes to the core transcription logic (`src/core`), and swapping out an external library requires changes primarily within `src/adapters`. This addresses potential :MaintainabilityIssues.

## Usage

### GUI Interface

Run the desktop application (entry point now located in the presentation layer):

```bash
python src/presentation/desktop_app.py
```

### Command Line Interface

Run the pipeline using the convenience script:

```bash
run_pipeline.bat <playlist_url> [options]
```

*(Note: While the script name remains, its internal calls may have been updated to reflect the new structure, likely invoking functionality within `src/presentation` or `src/core`.)*

## Requirements

- Python 3.9+ (:Technology choice)
- FFmpeg (for audio/video processing)
- CUDA-compatible GPU (recommended for faster WhisperX/Pyannote processing)
- OpenRouter API key (for LLM-based correction and summarization) - see `config/`
- Hugging Face token (for pyannote.audio models) - see `config/`

## Installation

1.  Clone the repository: `git clone <repository_url>`
2.  Navigate to the project directory: `cd <project_directory>`
3.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  Install dependencies: `pip install -r requirements.txt`
5.  Set up environment variables: Copy `.env.example` to `.env` and fill in your API keys/tokens.
    ```env
    # .env
    OPENROUTER_API_KEY=your_api_key
    HF_TOKEN=your_huggingface_token
    ```
6.  Install FFmpeg and ensure it's in your system PATH.

## Testing

The `tests/` directory contains automated tests. Run them using a test runner like `pytest`:

```bash
pytest tests/
```
This executes both unit tests for individual components and integration tests, contributing to the project's overall **cumulative testing** strategy, ensuring that new changes don't break existing functionality.

## License

License information needs to be added here.

# YouTube Video Pipeline

A comprehensive tool for downloading, transcribing, and summarizing YouTube videos with speaker diarization.

## Features

- Download videos from YouTube playlists
- Transcribe audio using WhisperX (self-hosted)
- Perform speaker diarization using pyannote.audio
- Correct and format transcripts using LLMs via OpenRouter
- Generate summaries of video content
- User-friendly GUI interface

## Project Structure

```text
.
├── config/                 # Configuration files
├── docs/                   # Documentation
├── logs/                   # Log files
├── output/                 # Output directory for processed videos
├── sample/                 # Sample files for testing
├── src/                    # Source code
│   ├── gui/                # GUI-specific backend modules
│   ├── transcription/      # Transcription modules
│   ├── utils/              # Utility modules
│   ├── run_full_pipeline.py # Main pipeline script
│   └── youtube_direct.py   # YouTube interaction module
├── tests/                  # Test files
├── tools/                  # Utility scripts
├── whisperx/               # WhisperX source code
├── desktop_app.py          # GUI application
├── python_wrapper.py       # Python environment wrapper
├── requirements.txt        # Python dependencies
├── run_pipeline.bat        # CLI entry point for Windows
└── verify_structure.bat    # Structure verification script
```

## Usage

### GUI Interface

Run the desktop application:

```bash
python desktop_app.py
```

### Command Line Interface

Run the pipeline from the command line:

```bash
run_pipeline.bat <playlist_url> [options]
```

Or directly:

```bash
python python_wrapper.py pipeline <playlist_url> [options]
```

## Requirements

- Python 3.9+
- FFmpeg
- CUDA-compatible GPU (recommended for faster processing)
- OpenRouter API key (for LLM-based correction and summarization)
- Hugging Face token (for pyannote.audio models)

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file:

   ```env
   OPENROUTER_API_KEY=your_api_key
   HF_TOKEN=your_huggingface_token
   ```

4. Install FFmpeg and ensure it's in your system PATH

## License

License information

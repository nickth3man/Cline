# YouTube Video Processing Pipeline Usage Guide

## Overview

This pipeline automates the process of downloading, transcribing, correcting, summarizing, and diarizing YouTube videos and playlists. It uses OpenAI Whisper API for transcription and OpenRouter LLM for correction, summarization, and pseudo-diarization.

## Features

- **Audio Extraction**: Downloads and converts audio from YouTube videos to WAV format.
- **Transcription**: Uses OpenAI Whisper API for high-quality speech-to-text conversion.
- **Sentence Splitting**: Applies SpaCy for natural language processing to split transcripts into sentences.
- **Pseudo-Diarization**: Uses OpenRouter LLM to assign speaker labels based on context.
- **Transcript Correction**: Applies LLM-based correction to improve transcript quality.
- **Summarization**: Generates concise summaries of video content.
- **HTML Reader**: Creates a simple HTML interface for reading transcripts.
- **Progress Tracking**: Shows real-time progress and generates detailed statistics.
- **Error Handling**: Gracefully handles errors and continues processing other videos.

## Requirements

- Python 3.8+
- Virtual environment (venv)
- Required API keys:
  - `OPENAI_API_KEY`: For Whisper transcription
  - `OPENROUTER_API_KEY`: For LLM-based tasks
  - `HF_TOKEN` (optional): For Hugging Face models

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys (see Environment Variables section)

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=sk-...your-openai-api-key...
OPENROUTER_API_KEY=sk-or-v1-...your-openrouter-api-key...
HF_TOKEN=hf_...your-huggingface-token...  # Optional
```

## Usage

### Processing a YouTube Playlist

```bash
# Activate the virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run the pipeline with a YouTube playlist URL
python -m src.run_full_pipeline https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID
```

### Command-line Arguments

The pipeline accepts the following command-line arguments:

- Playlist URL (required): URL of the YouTube playlist to process
- Output directory (optional, default: "output"): Directory to save processed files
- Correction models (optional, default: ["openai/gpt-4.1-mini", "anthropic/claude-3-haiku-20240307"]): LLM models for transcript correction
- Summarization model (optional, default: "openai/gpt-4.1-mini"): LLM model for summarization

Example with custom options:
```bash
python -m src.run_full_pipeline https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID output_custom anthropic/claude-3-opus-20240229 anthropic/claude-3-opus-20240229
```

## Output Structure

For each video, the pipeline creates a folder with the following files:

- `{title}_audio.wav`: Extracted audio in WAV format
- `{title}_raw_transcript.txt`: Raw transcript from Whisper
- `{title}_diarization.json`: Speaker diarization results
- `{title}_corrected_transcript_{model}.txt`: Corrected transcript
- `{title}_summary_{model}.txt`: Summary of the video content
- `{title}_transcript_reader.html`: HTML interface for reading the transcript
- `{title}_metadata.json`: Metadata about the video and processing
- `{title}_stats.json`: Statistics about the processing time and steps

Additionally, the pipeline generates:
- `pipeline_stats.json`: Overall statistics for the entire playlist processing
- `pipeline.log`: Detailed log of the processing steps and any errors

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure you've activated the virtual environment and installed all dependencies.
   ```bash
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **API Key Errors**: Check that your `.env` file contains valid API keys.
   - OpenAI API key should start with `sk-` (not `sk-proj-` or `ssk-proj-`)
   - OpenRouter API key should start with `sk-or-v1-`

3. **YouTube Download Errors**: Some videos may be unavailable, private, or region-locked. The pipeline will skip these and continue with available videos.

4. **Memory Issues**: For very long videos, you may need more RAM. Consider processing fewer videos at once.

### Error Logs

Check the `pipeline.log` file for detailed error messages and troubleshooting information.

## Performance Optimization

- Processing time varies based on video length and model selection.
- For faster processing, use smaller/faster models like `anthropic/claude-3-haiku` or `openai/gpt-4.1-mini`.
- For higher quality, use larger models like `anthropic/claude-3-opus` or `openai/gpt-4.1-turbo`.

## Advanced Configuration

For advanced configuration options, modify the following files:

- `src/run_full_pipeline.py`: Main pipeline logic and workflow
- `src/transcription/transcription_workflow.py`: Transcription and LLM processing settings
- `src/utils/workflow_logic.py`: Utility functions for the pipeline

## License

[Your License Information]

## Contact

[Your Contact Information]

---

## Dependency Check Script

To help ensure your environment is correctly set up before running the pipeline, a dependency check script is provided.

### Purpose

- Verifies that all required Python packages are installed.
- Checks that FFmpeg is installed and accessible.
- Confirms the SpaCy language model is downloaded.
- Checks that the Hugging Face token (`HF_TOKEN`) is set in your environment variables.

### Usage

Run the script from the project root:

```bash
python scripts/check_dependencies.py
```

The script will print a report of any missing dependencies and instructions for resolving them. It exits with code 0 if all dependencies are satisfied, or 1 if issues are found.

### Benefits

- Early detection of missing or misconfigured dependencies.
- Clear guidance on how to fix environment setup issues.
- Helps avoid runtime errors during pipeline execution.

Please run this script after setting up your environment and before running the pipeline to ensure a smooth experience.

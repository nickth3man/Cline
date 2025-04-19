# GUI Backend Modules

This directory contains backend modules that support the desktop GUI application (`desktop_app.py`).

## Modules

- `download_handler.py`: Handles downloading videos from YouTube playlists
- `summarize_handler.py`: Processes and summarizes transcripts using OpenRouter LLMs

## Integration with GUI

These modules are called by the desktop application through subprocess calls. They provide structured output that the GUI can parse to update the user interface.

Example of structured output format:
- `VIDEO_STATUS:{video_id}:{status_message}` - Updates status for a specific video
- `VIDEO_ERROR:{video_id}:{error_message}` - Reports an error for a specific video

## Usage

While these modules are primarily designed to be called by the GUI, they can also be run directly from the command line:

```bash
python src/gui/download_handler.py --playlist_url "https://www.youtube.com/playlist?list=XXXX" --output_path "output"
python src/gui/summarize_handler.py --video_ids "video1,video2" --model_name "openai/gpt-4.1-mini" --output_path "output"
```

# --- START OF FILE README.md ---

# YouTube Playlist Advanced Processor

This application downloads videos from a YouTube playlist, extracts audio, performs transcription (Whisper via OpenRouter), speaker diarization (local pyannote.audio), and corrects/formats the transcript using an LLM (via OpenRouter).

## Features

- Download videos in multiple resolutions.
- Extract audio and transcribe it.
- Generate summaries of transcriptions.
- Group all assets for each video in a single folder.
- **Data Export:** Export metadata, transcripts, and summaries to CSV or JSON Lines format via CLI and GUI.

## Prerequisites

1.  **Python:** Python 3.9 or higher recommended.
2.  **FFmpeg:** Must be installed and accessible in your system's PATH. FFmpeg is used for audio extraction and conversion. Download from [https://ffmpeg.org/](https://ffmpeg.org/).
3.  **Git:** (Optional) For cloning the repository.

## Setup

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `torch` and `torchaudio` might take time. Ensure you have sufficient disk space.*

4.  **Configure API Keys and Tokens:**
    *   Create a file named `.env` in the project's root directory (where `main.py` is located).
    *   Add your OpenRouter API key to the `.env` file:
        ```dotenv
        OPENROUTER_API_KEY="sk-or-v1-your-openrouter-api-key-here"
        ```
    *   **(Optional but Recommended for `pyannote.audio`)** Add your Hugging Face Hub token (if needed for the diarization model, e.g., `pyannote/speaker-diarization-3.1`). You need to accept the model's terms on the Hugging Face website first. Get a token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
        ```dotenv
        HF_TOKEN="hf_your_huggingface_token_here"
        ```
        Alternatively, you can log in using `huggingface-cli login` in your terminal before running the application.

5.  **Accept `pyannote.audio` Model Terms:**
    *   You MUST manually visit the Hugging Face Hub pages for the models used by the `pyannote/speaker-diarization-3.1` pipeline (check the `pyannote.audio` documentation or error messages for specific models like `pyannote/segmentation-3.0` and `speechbrain/speaker-recognition-ecapa-tdnn`) and accept their terms of use while logged into your Hugging Face account. The application cannot do this for you. Failure to do so will likely result in errors when loading the diarization pipeline.

## Running the Application

1.  Ensure your virtual environment is active.
2.  Ensure the `.env` file is configured correctly.
3.  Run the main script:
    ```bash
    python main.py
    ```
4.  Use the GUI to:
    *   Enter the YouTube Playlist URL.
    *   Set the maximum video resolution.
    *   Choose the LLM for transcript correction.
    *   Select the main output directory.
    *   Click "Start Processing".

## Database Management & Export (CLI & GUI)

The application uses an SQLite database (`pipeline_output.db` by default, located in the `--work_dir`) to track video status, metadata, and generated content (transcripts, summaries). You can manage and export this data using either the GUI or the command-line script (`transcription_workflow.py`).

**GUI Management ("Manage Data" Tab):**

*   **Refresh Data:** Reloads the video list from the database.
*   **Retry All Errors:** Marks all videos with an 'error_*' status for reprocessing the next time the pipeline runs.
*   **Context Menu (Right-Click on Table Rows):**
    *   **View Details:** Shows detailed information for the selected video (single selection only).
    *   **Retry Selected:** Marks the selected video(s) for reprocessing.
    *   **Delete Selected...:** Deletes the selected video record(s) from the database. Prompts whether to also delete associated files/folders from disk.
    *   **Export Selected...:** Exports data (metadata, transcript, summary) for the selected video(s) to a CSV or JSON Lines file. Prompts for save location and format. Export runs in the background.

**CLI Management (`transcription_workflow.py`):**

*   Run the script with management flags instead of `--playlist_url`. Specify the `--work_dir` if it's not the default `yt_pipeline_output`.
*   `--status`: Display a summary table of all videos in the database.
*   `--list-errors`: List only videos with an 'error_*' status.
*   `--view VIDEO_ID`: Show detailed information for a specific video ID.
*   `--retry VIDEO_ID|all-errors`: Mark a specific video ID or all videos with errors for reprocessing (sets status to `downloaded` or `pending_download`).
*   `--delete VIDEO_ID`: Delete a video record from the database (prompts for file deletion).
*   `--export FILEPATH [--ids ID1 ID2 ...] [--status STATUS]`: Export data to `FILEPATH`.
    *   Format (`.csv` or `.jsonl`) is determined by the `FILEPATH` extension.
    *   Optionally filter by specific `--ids` (space-separated) or `--status` (e.g., `processed`, `error_%`).

    *Example: Export processed videos with specific IDs to CSV*
    ```bash
    python transcription_workflow.py --work_dir yt_pipeline_output --export processed_subset.csv --status processed --ids videoId1 videoId2
    ```

## Running Tests

1.  Ensure your virtual environment is active.
2.  Install test dependencies (already included in `requirements.txt`):
    ```bash
    pip install pytest pytest-mock
    ```
3.  Run pytest from the project's root directory:
    ```bash
    pytest
    ```

## Important Notes

*   **API Costs:** Using OpenRouter incurs costs based on the models selected (Whisper, Correction LLM) and the amount of processing. Monitor your usage and costs on the OpenRouter website.
*   **Processing Time:** Processing can be lengthy, depending on the number/duration of videos, network speed, API response times, and your computer's performance (especially for local diarization).
*   **`pyannote.audio`:** Diarization quality can vary. It requires downloading models on the first run, which can take time and requires internet access. A GPU significantly speeds up diarization.
*   **Error Handling:** The application attempts to handle errors per video, allowing the rest of the playlist to process. Check the log/error panel in the GUI and the console output for details if issues occur.
*   **Stopping:** Clicking "Stop" signals the current video to finish its *current major step* (Download, Convert, Transcribe, Diarize, Correct) before stopping. It may not be instantaneous.

# --- END OF FILE README.md ---

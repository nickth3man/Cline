"""
Download and process videos from a YouTube playlist.

This script handles downloading videos from a YouTube playlist and ensures
the output structure follows the project's strict requirements.

Usage:
    python download_and_process.py [youtube_playlist_url]
"""

import os
import sys
import logging
import datetime
import subprocess
import json
import shutil
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = (
    log_dir
    / f"download_process_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)


def ensure_output_dir():
    """Ensure the output directory exists and create it if not."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def enforce_output_structure(video_dir, video_filename):
    """
    Enforces the strict output structure rule: each video folder should only contain
    1. The original video file
    2. The transcript from the video
    3. The transcript summary

    Args:
        video_dir: Path to the video directory
        video_filename: Filename of the original video (without path)
    """
    video_dir = Path(video_dir)
    base_name = video_filename.rsplit(".", 1)[0]

    # List of allowed files
    allowed_files = {
        video_filename,  # Original video
        f"{base_name}_corrected.md",  # Transcript
        f"{base_name}_summary.md",  # Summary
    }

    # Remove any files that don't match the allowed pattern
    for file in video_dir.iterdir():
        if file.is_file() and file.name not in allowed_files:
            file.unlink()
        elif file.is_dir():
            shutil.rmtree(file)

    logging.info(f"Enforced structure for {video_dir.name}")


def download_playlist(playlist_url):
    """
    Download videos from a YouTube playlist using yt-dlp.

    Args:
        playlist_url: URL of the YouTube playlist

    Returns:
        list: List of downloaded video paths
    """
    temp_dir = Path("temp_downloads")
    temp_dir.mkdir(exist_ok=True)

    logging.info(f"Downloading playlist: {playlist_url}")
    logging.debug(f"Script sys.path: {sys.path}")
    logging.debug(f"Script PATH env var: {os.environ.get('PATH')}")
    logging.debug(f"Script VIRTUAL_ENV env var: {os.environ.get('VIRTUAL_ENV')}")

    # Basic yt-dlp command with format selection
    cmd = [
        "yt-dlp",
        "--ignore-errors",
        "--format",
        "best[ext=mp4]/best",  # Try best MP4, fallback to best available
        "--output",
        str(temp_dir / "%(title)s.%(ext)s"),
        "--no-playlist-reverse",
        "--restrict-filenames",
        playlist_url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.debug(f"yt-dlp stdout:\n{result.stdout}")
        if result.stderr:
            logging.debug(f"yt-dlp stderr:\n{result.stderr}")

        # Get list of downloaded files
        downloaded_files = []
        for file in temp_dir.iterdir():
            if file.is_file() and not file.name.endswith(".part"):
                downloaded_files.append(file)

        return downloaded_files

    except subprocess.CalledProcessError as e:
        logging.error(
            f"yt-dlp command failed with return code {e.returncode}. Stderr:\n{e.stderr}"
        )
        logging.error(f"Subprocess Environment PATH: {os.environ.get('PATH')}")
        return []


def main():
    """Main function to download and process a YouTube playlist."""
    logging.info("Starting download and processing script")
    logging.debug(f"Initial sys.executable: {sys.executable}")
    logging.debug(f"Initial sys.path: {sys.path}")
    logging.debug(f"Initial PATH env var: {os.environ.get('PATH')}")
    logging.debug(f"Initial VIRTUAL_ENV env var: {os.environ.get('VIRTUAL_ENV')}")

    if len(sys.argv) < 2:
        print("Please provide a YouTube playlist URL")
        sys.exit(1)

    playlist_url = sys.argv[1]
    output_dir = ensure_output_dir()

    # Download videos
    downloaded_files = download_playlist(playlist_url)

    if not downloaded_files:
        logging.error("No videos were downloaded successfully")
        sys.exit(1)

    # Process each downloaded video
    for video_file in downloaded_files:
        video_name = video_file.stem
        video_dir = output_dir / video_name
        video_dir.mkdir(exist_ok=True)

        # Move video to its output directory
        target_path = video_dir / video_file.name
        shutil.move(str(video_file), str(target_path))

        # Enforce output structure
        enforce_output_structure(video_dir, video_file.name)

    logging.info("Download and processing completed successfully")


if __name__ == "__main__":
    main()

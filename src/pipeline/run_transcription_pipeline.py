#!/usr/bin/env python3
"""
Run the transcription pipeline on already downloaded videos.

This script processes videos that have already been downloaded and organizes them
according to the project's strict output structure requirements.
"""
import os
import sys
import logging
import datetime
import argparse
from pathlib import Path
import shutil

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import project modules
from transcription.transcription_workflow import (
    extract_or_convert_audio,
    transcribe_audio,
    diarize_speakers,
    correct_transcript,
    summarize_transcript,
)
from utils.workflow_logic import enforce_output_structure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = (
    log_dir
    / f"transcription_pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)


def process_video(
    video_path, output_dir, transcription_model=None, correction_model=None
):
    """
    Process a single video through the transcription pipeline.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the processed output
        transcription_model: Optional model to use for transcription
        correction_model: Optional model to use for correction

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        video_path = Path(video_path)
        if not video_path.exists():
            logging.error(f"Video file not found: {video_path}")
            return False

        # Create output directory for this video
        video_dir = Path(output_dir) / video_path.stem
        video_dir.mkdir(exist_ok=True, parents=True)

        # Copy the video file to the output directory
        output_video_path = video_dir / video_path.name
        if not output_video_path.exists():
            shutil.copy2(video_path, output_video_path)
            logging.info(f"Copied video to: {output_video_path}")

        # Extract audio from video
        audio_path = video_dir / f"{video_path.stem}.wav"
        extract_or_convert_audio(str(video_path), str(audio_path))
        logging.info(f"Extracted audio to: {audio_path}")

        # Transcribe audio
        logging.info(f"Transcribing audio: {audio_path}")
        raw_transcript = transcribe_audio(str(audio_path))

        # Diarize speakers
        logging.info(f"Performing speaker diarization")
        diarization_result = diarize_speakers(str(audio_path), raw_transcript)

        # Correct transcript
        logging.info(f"Correcting transcript")
        corrected_transcript = correct_transcript(
            raw_transcript, diarization_result, correction_model
        )

        # Save corrected transcript
        transcript_path = video_dir / f"{video_path.stem}_corrected.md"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(corrected_transcript)
        logging.info(f"Saved corrected transcript to: {transcript_path}")

        # Summarize transcript
        logging.info(f"Generating summary")
        summary = summarize_transcript(corrected_transcript)

        # Save summary
        summary_path = video_dir / f"{video_path.stem}_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        logging.info(f"Saved summary to: {summary_path}")

        # Enforce output structure
        logging.info(f"Enforcing output structure")
        if enforce_output_structure(str(video_dir), video_path.name):
            logging.info(f"Successfully processed video: {video_path.name}")
            return True
        else:
            logging.error(f"Failed to enforce output structure for: {video_path.name}")
            return False

    except Exception as e:
        logging.error(f"Error processing video {video_path}: {e}", exc_info=True)
        return False


def main():
    """Main function to process videos through the transcription pipeline."""
    parser = argparse.ArgumentParser(
        description="Process downloaded videos through the transcription pipeline"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="temp_downloads",
        help="Directory containing downloaded videos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save processed videos",
    )
    parser.add_argument(
        "--transcription_model",
        type=str,
        default=None,
        help="Model to use for transcription",
    )
    parser.add_argument(
        "--correction_model",
        type=str,
        default="anthropic/claude-3-haiku-20240307",
        help="Model to use for transcript correction",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        return 1

    output_dir.mkdir(exist_ok=True)

    # Get list of video files
    video_files = list(input_dir.glob("*.mp4"))
    video_files = [f for f in video_files if not f.name.endswith(".part")]

    if not video_files:
        logging.error(f"No video files found in: {input_dir}")
        return 1

    logging.info(f"Found {len(video_files)} videos to process")

    success_count = 0
    failure_count = 0

    for video_file in video_files:
        if process_video(
            video_file, output_dir, args.transcription_model, args.correction_model
        ):
            success_count += 1
        else:
            failure_count += 1

    logging.info(
        f"Processing complete: {success_count} videos processed successfully, {failure_count} failed"
    )

    if failure_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

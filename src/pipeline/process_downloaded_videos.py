"""
Process downloaded videos and ensure they follow the project's strict output structure.

This script processes videos that have already been downloaded and organizes them
according to the project's strict output structure requirements.
"""

import os
import sys
import logging
import datetime
import shutil
from pathlib import Path

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import project modules
from utils.workflow_logic import (
    extract_or_convert_audio,
    transcribe_audio,
    diarize_speakers,
    correct_transcript,
    enforce_output_structure,
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = (
    log_dir / f"process_videos_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
)


def process_downloaded_videos():
    """
    Process downloaded videos from temp_downloads directory and move them to output directory
    with the proper structure.
    """
    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Get temp downloads directory
    temp_dir = Path("temp_downloads")
    if not temp_dir.exists() or not temp_dir.is_dir():
        logging.error(f"Temp downloads directory not found: {temp_dir}")
        return False

    # Get list of completed video downloads (exclude .part and .ytdl files)
    video_files = [f for f in temp_dir.glob("*.mp4") if not f.name.endswith(".part")]

    if not video_files:
        logging.warning(
            "No completed video downloads found in temp_downloads directory"
        )
        return False

    logging.info(f"Found {len(video_files)} completed video downloads to process")

    success_count = 0
    failure_count = 0

    for video_file in video_files:
        try:
            # Create a directory for the video in output
            video_dir = output_dir / video_file.stem
            video_dir.mkdir(exist_ok=True)

            # Move the video file to its directory
            target_path = video_dir / video_file.name
            logging.info(f"Moving {video_file} to {target_path}")
            shutil.copy2(video_file, target_path)

            # Extract audio from video
            audio_path = video_dir / f"{video_file.stem}.wav"
            extract_or_convert_audio(str(video_file), str(audio_path))
            logging.info(f"Extracted audio to: {audio_path}")

            # Transcribe audio
            logging.info(f"Transcribing audio: {audio_path}")
            raw_transcript = transcribe_audio(str(audio_path))

            # Diarize speakers
            logging.info(f"Performing speaker diarization")
            diarization_result = diarize_speakers(str(audio_path))

            # Correct transcript
            logging.info(f"Correcting transcript")
            corrected_transcript = correct_transcript(
                raw_transcript, diarization_result, "anthropic/claude-3-haiku-20240307"
            )

            # Save corrected transcript
            transcript_path = video_dir / f"{video_file.stem}_corrected.md"
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(corrected_transcript)
            logging.info(f"Saved corrected transcript to: {transcript_path}")

            # Generate summary using the same model
            logging.info(f"Generating summary")
            from openai import OpenAI

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

            summary_prompt = f"""You are an expert at summarizing content. Create a concise but comprehensive summary of the following transcript. 
            Focus on the key points, main arguments, and important conclusions. The summary should be well-structured with clear headings 
            and bullet points where appropriate.

            TRANSCRIPT:
            {corrected_transcript}

            SUMMARY:
            """

            chat_completion = client.chat.completions.create(
                model="anthropic/claude-3-haiku-20240307",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert content summarizer.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.5,
            )

            summary = chat_completion.choices[0].message.content.strip()

            # Save summary
            summary_path = video_dir / f"{video_file.stem}_summary.md"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            logging.info(f"Saved summary to: {summary_path}")

            # Enforce the output structure
            if enforce_output_structure(str(video_dir), video_file.name):
                success_count += 1
                logging.info(f"Successfully processed video: {video_file.name}")
            else:
                failure_count += 1
                logging.error(
                    f"Failed to enforce output structure for: {video_file.name}"
                )

        except Exception as e:
            failure_count += 1
            logging.error(
                f"Error processing video {video_file.name}: {e}", exc_info=True
            )

    logging.info(
        f"Processing complete: {success_count} videos processed successfully, {failure_count} failed"
    )
    return success_count > 0


if __name__ == "__main__":
    logging.info("Starting processing of downloaded videos")

    if process_downloaded_videos():
        logging.info("Video processing completed successfully")
        sys.exit(0)
    else:
        logging.error("Video processing failed")
        sys.exit(1)

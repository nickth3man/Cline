import os
import json
import logging
import re
import yt_dlp
import datetime
import time
import argparse # Import argparse
from tqdm import tqdm
from src.utils.workflow_logic import (
    transcribe_audio,
    # Removed perform_diarization_and_assignment as it's now part of transcribe_audio
    correct_transcript,
    summarize_transcript, # Import summarize_transcript
)
from src.utils import workflow_logic # Keep this import for other potential workflow_logic functions
from src.youtube_direct import YouTubeDirectHandler

# Configure logging with timestamps and levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "..", "pipeline.log")
        ),
    ],
)


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r"[^\w\-. ]", "", filename)  # Allow dots and hyphens
    sanitized = re.sub(
        r"\s+", "_", sanitized
    )  # Replace whitespace sequences with underscore
    return sanitized[:150]  # Limit length


def generate_html_reader(corrected_transcript: str, output_path: str):
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Transcript Reader</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; }}
</style>
</head>
<body>
<h1>Transcript Reader</h1>
<pre>{corrected_transcript}</pre>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logging.info(f"HTML transcript reader saved: {output_path}")


def run_pipeline(
    playlist_url: str,
    output_base_dir: str,
    correction_models: list,
    summarization_model: str,
    video_ids: list = None, # Add video_ids parameter with default None
):
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Track overall statistics
    stats = {
        "total_videos": 0,
        "processed_videos": 0,
        "skipped_videos": 0,
        "failed_videos": 0,
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": None,
        "total_duration_seconds": 0,
        "playlist_url": playlist_url,
        "correction_models": correction_models,
        "summarization_model": summarization_model,
    }

    try:
        # First try our direct YouTube handler (more robust method)
        logging.info(f"Fetching playlist videos using direct handler: {playlist_url}")
        try:
            youtube_handler = YouTubeDirectHandler()
            direct_video_list = youtube_handler.get_playlist_videos(playlist_url)
            if direct_video_list:
                logging.info(
                    f"Successfully extracted {len(direct_video_list)} videos with direct handler"
                )
                video_list = direct_video_list
            else:
                raise ValueError("No videos found with direct handler")
        except Exception as e:
            logging.warning(f"Direct handler failed: {e}. Falling back to yt-dlp...")

            # Fallback to standard yt-dlp extraction
            ydl_opts_info = {
                "quiet": True,
                "extract_flat": "in_playlist",
                "skip_download": True,
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                },
            }

            # Track video extraction progress for debugging
            logging.info(
                f"Fetching video list from playlist using yt-dlp: {playlist_url}"
            )
            video_list = []
            # First try standard extraction
            with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                try:
                    info = ydl.extract_info(playlist_url, download=False)
                    if "entries" in info and info["entries"]:
                        video_list = info["entries"]
                        logging.info(
                            f"Successfully extracted {len(video_list)} videos from playlist"
                        )
                except Exception as e:
                    logging.warning(
                        f"Standard playlist extraction failed: {e}. Trying alternate method..."
                    )

            # If that didn't work, try direct approach with direct URL construction
            if not video_list:
                logging.info("Attempting direct playlist item extraction...")
                # Get playlist ID - handles both full URLs and playlist IDs
                playlist_id = playlist_url.split("list=")[-1].split("&")[0]

                # Remove placeholder video URL usage; instead, try to extract videos via yt-dlp playlist extraction only
                # We will not attempt individual video lookup with placeholder video ID

                # Try to extract playlist videos again with yt-dlp, but with different options
                ydl_opts_playlist = {
                    "quiet": True,
                    "extract_flat": True,
                    "skip_download": True,
                    "forceurl": True,
                    "http_headers": {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                    },
                }
                try:
                    with yt_dlp.YoutubeDL(ydl_opts_playlist) as ydl:
                        info = ydl.extract_info(playlist_url, download=False)
                        if "entries" in info and info["entries"]:
                            video_list = info["entries"]
                            logging.info(
                                f"Playlist extraction succeeded with {len(video_list)} videos on retry"
                            )
                except Exception as e:
                    logging.error(f"Retry playlist extraction failed: {e}")

            if not video_list:
                raise ValueError(
                    "Could not extract any videos from the playlist after multiple attempts"
                )

        stats["total_videos"] = len(video_list)
        logging.info(f"Found {len(video_list)} videos in playlist.")

        # Filter video_list if video_ids are provided
        if video_ids:
            original_video_list = video_list # Keep original for total count
            video_list = [video for video in video_list if video.get("id") in video_ids]
            logging.info(f"Filtered to {len(video_list)} videos based on provided IDs.")
            stats["total_videos"] = len(video_list) # Update total videos to process

    except Exception as e:
        logging.error(f"Failed to fetch playlist info: {e}", exc_info=True)
        raise RuntimeError(f"Playlist fetching failed: {e}")

    # Add total duration to stats if available from yt-dlp info
    # Note: 'info' might not be defined if the direct handler succeeded.
    # We should only access 'info' if the fallback to yt-dlp occurred.
    # A more robust approach would be to get duration from each video entry if available.
    # For now, we'll leave this as is, acknowledging it might not always set total_duration_seconds.
    # if 'duration' in info:
    #     stats['total_duration_seconds'] = info['duration']


    # Create a progress bar for the overall process
    pbar = tqdm(total=len(video_list), desc="Processing videos", unit="video")

    # Process each video
    for index, video in enumerate(video_list):
        start_time = time.time()
        video_title = video.get("title", f"video_{index}")
        video_id = video.get("id", f"unknown_id_{index}") # Get video ID
        video_url = video.get("url") or video.get("webpage_url") or video.get("id")

        print(f"VIDEO_STATUS:{video_id}:Starting processing for video: {video_title}") # Structured status

        # Update progress bar description
        pbar.set_description(f"Processing: {video_title[:30]}...")

        # Skip if no valid URL is found (e.g., deleted video)
        if (
            not video_url
            or "unavailable" in video_title.lower()
            or "deleted" in video_title.lower()
        ):
            warning_msg = f"Skipping video {index+1}/{len(video_list)} ('{video_title}') due to missing URL or unavailable status."
            logging.warning(warning_msg)
            print(f"VIDEO_STATUS:{video_id}:{warning_msg}") # Structured status for skipped
            stats["skipped_videos"] += 1
            pbar.update(1)
            continue

        sanitized_title = sanitize_filename(video_title)
        # Use only the title for the folder name (no index prefix)
        video_folder = os.path.join(output_base_dir, sanitized_title)
        os.makedirs(video_folder, exist_ok=True)

        logging.info(f"Processing video {index+1}/{len(video_list)}: {video_title}")

        # Track video-specific stats
        video_stats = {
            "title": video_title,
            "url": video_url,
            "index": index,
            "start_time": datetime.datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": 0,
            "success": False,
            "steps_completed": [],
            "error": None,
        }

        # Download and convert audio directly to WAV using yt-dlp postprocessor
        wav_audio_path = os.path.join(video_folder, f"{sanitized_title}_audio.wav")
        ydl_opts_download = {
            "format": "bestaudio/best",
            "outtmpl": wav_audio_path.replace(".wav", ""),  # yt-dlp adds extension
            "quiet": True,
            "noprogress": True,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            },
            "socket_timeout": 30,
            "retries": 5,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",  # Lossless WAV
                }
            ],
        }

        audio_downloaded = False
        for attempt in range(3):  # Try up to 3 times with different methods
            try:
                status_msg = f"Downloading audio (attempt {attempt+1}/3): {video_url}"
                logging.info(status_msg)
                print(f"VIDEO_STATUS:{video_id}:Status: {status_msg}") # Structured status

                # Create attempt-specific path to avoid file locking issues
                attempt_audio_path = wav_audio_path.replace(
                    ".wav", f"_attempt{attempt+1}.wav"
                )

                if attempt == 0:
                    # Standard method
                    modified_opts = ydl_opts_download.copy()
                    modified_opts["outtmpl"] = attempt_audio_path.replace(
                        ".wav", ""
                    )  # yt-dlp adds extension
                    with yt_dlp.YoutubeDL(modified_opts) as ydl_download:
                        ydl_download.download([video_url])
                elif attempt == 1:
                    # Try different format selection
                    modified_opts = ydl_opts_download.copy()
                    modified_opts["outtmpl"] = attempt_audio_path.replace(
                        ".wav", ""
                    )  # yt-dlp adds extension
                    modified_opts["format"] = "bestaudio[ext=m4a]/bestaudio/best"
                    with yt_dlp.YoutubeDL(modified_opts) as ydl_download:
                        ydl_download.download([video_url])
                else:
                    # Last resort: try with different user agent and options
                    modified_opts = ydl_opts_download.copy()
                    modified_opts["outtmpl"] = attempt_audio_path.replace(
                        ".wav", ""
                    )  # yt-dlp adds extension
                    modified_opts["http_headers"] = {
                        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"
                    }
                    modified_opts["format"] = (
                        "worstaudio/worst"  # Even get low quality if needed
                    )
                    try:
                        modified_opts["external_downloader"] = "aria2c"
                    except:
                        logging.warning(
                            "External downloader aria2c not available, using default"
                        )
                    with yt_dlp.YoutubeDL(modified_opts) as ydl_download:
                        ydl_download.download([video_url])

                # Check if the attempt-specific file exists after download
                if os.path.exists(attempt_audio_path):
                    # If successful, rename to the standard path (ensuring no file lock)
                    if os.path.exists(wav_audio_path):
                        try:
                            os.remove(wav_audio_path)
                        except Exception as e:
                            logging.warning(
                                f"Could not remove existing file {wav_audio_path}: {e}"
                            )
                            # Try an alternative name instead
                            wav_audio_path = os.path.join(
                                video_folder, f"{sanitized_title}_audio_final.wav"
                            )

                    # Give the file system a moment to release any locks
                    time.sleep(1)

                    try:
                        import shutil

                        shutil.copy2(attempt_audio_path, wav_audio_path)
                        os.remove(attempt_audio_path)  # Clean up the attempt file
                        audio_downloaded = True
                        status_msg = f"Successfully downloaded audio on attempt {attempt+1}"
                        logging.info(status_msg)
                        print(f"VIDEO_STATUS:{video_id}:Status: {status_msg}") # Structured status
                        break
                    except Exception as e:
                        warning_msg = f"Could not rename file {attempt_audio_path} to {wav_audio_path}: {e}"
                        logging.warning(warning_msg)
                        print(f"VIDEO_ERROR:{video_id}:Error: {warning_msg}") # Structured error
                        # Use the attempt file directly if we can't rename
                        wav_audio_path = attempt_audio_path
                        audio_downloaded = True
                        status_msg = f"Using attempt file directly: {wav_audio_path}"
                        logging.info(status_msg)
                        print(f"VIDEO_STATUS:{video_id}:Status: {status_msg}") # Structured status
                        break

                # Sometimes yt-dlp saves with a different extension (.m4a, .webm etc.) before converting
                # Let's find the actual audio file downloaded if wav isn't there
                potential_files = [
                    f
                    for f in os.listdir(video_folder)
                    if f.startswith(sanitized_title + "_audio")
                    and f.endswith(f"_attempt{attempt+1}.wav")
                ]
                if not potential_files:
                    # Try without the attempt suffix
                    potential_files = [
                        f
                        for f in os.listdir(video_folder)
                        if f.startswith(sanitized_title + "_audio")
                    ]

                if potential_files:
                    actual_audio_file = os.path.join(video_folder, potential_files[0])
                    warning_msg = f"Expected WAV not found, using alternative: {actual_audio_file}"
                    logging.warning(warning_msg)
                    print(f"VIDEO_STATUS:{video_id}:Status: {warning_msg}") # Structured status (using status for non-critical issue)
                    wav_audio_path = actual_audio_file
                    audio_downloaded = True
                    break
                else:
                    warning_msg = f"No audio files found on attempt {attempt+1}, trying next method..."
                    logging.warning(warning_msg)
                    print(f"VIDEO_STATUS:{video_id}:Status: {warning_msg}") # Structured status (using status for non-critical issue)
                    # Give the file system time to release any locks before next attempt
                    time.sleep(2)

            except Exception as e:
                error_msg = f"Audio download failed on attempt {attempt+1}: {e}"
                logging.error(error_msg, exc_info=True)
                print(f"VIDEO_ERROR:{video_id}:Error: {error_msg}") # Structured error
                if attempt < 2:  # Only log warning if we have more attempts left
                    logging.info(f"Retrying with different download method...")
                    # Give the file system time to release any locks before next attempt
                    time.sleep(2)
                else:
                    final_error_msg = f"All audio download attempts failed for {video_title}"
                    logging.error(final_error_msg)
                    print(f"VIDEO_ERROR:{video_id}:Error: {final_error_msg}") # Structured error
                    video_stats["error"] = final_error_msg

        if not audio_downloaded:
            error_msg = f"Failed to download audio after multiple attempts for {video_title}"
            logging.error(error_msg)
            print(f"VIDEO_ERROR:{video_id}:Error: {error_msg}") # Structured error
            video_stats["end_time"] = datetime.datetime.now().isoformat()
            video_stats["duration_seconds"] = time.time() - start_time
            stats_path = os.path.join(video_folder, f"{sanitized_title}_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(video_stats, f, indent=2)
            stats["failed_videos"] += 1
            pbar.update(1)
            continue  # Skip to next video

        logging.info(f"Downloaded and converted audio to {wav_audio_path}")
        print(f"VIDEO_STATUS:{video_id}:Status: Audio downloaded and converted to {wav_audio_path}") # Structured status
        video_stats["steps_completed"].append("audio_download")

        # Transcribe audio (includes diarization now)
        print(f"VIDEO_STATUS:{video_id}:Starting transcription and diarization") # Structured status
        try:
            # transcribe_audio now returns the result with segments and speaker info
            transcription_diarization_result = transcribe_audio(video_id, wav_audio_path)
            segments_with_speakers = transcription_diarization_result.get("segments", [])
            raw_transcript_text = transcription_diarization_result.get("text", "") # Use the full text from the result

            raw_transcript_path = os.path.join(
                video_folder, f"{sanitized_title}_raw_transcript.txt"
            )
            with open(raw_transcript_path, "w", encoding="utf-8") as f:
                f.write(raw_transcript_text) # Write the extracted text
            logging.info(f"Raw transcript saved: {raw_transcript_path}")
            print(f"VIDEO_STATUS:{video_id}:Status: Raw transcript saved to {raw_transcript_path}") # Structured status
            video_stats["steps_completed"].append("transcription_and_diarization")

            # Save the diarization result (segments with speakers) to a JSON file
            diarization_path = os.path.join(
                video_folder, f"{sanitized_title}_diarization.json"
            )
            with open(diarization_path, "w", encoding="utf-8") as f:
                json.dump(segments_with_speakers, f, indent=2)
            logging.info(f"Diarization results saved: {diarization_path}")
            print(f"VIDEO_STATUS:{video_id}:Status: Diarization results saved to {diarization_path}") # Structured status


        except Exception as e:
            error_msg = f"Transcription or diarization failed for {video_title}: {e}"
            logging.error(error_msg, exc_info=True)
            print(f"VIDEO_ERROR:{video_id}:Error: {error_msg}") # Structured error
            video_stats["error"] = error_msg
            stats["failed_videos"] += 1

            # Save video stats even if failed
            video_stats["end_time"] = datetime.datetime.now().isoformat()
            video_stats["duration_seconds"] = time.time() - start_time
            stats_path = os.path.join(video_folder, f"{sanitized_title}_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(video_stats, f, indent=2)

            pbar.update(1)
            continue  # Skip to next video

        # Delete audio file after successful transcription/diarization
        if os.path.exists(wav_audio_path):
            try:
                os.remove(wav_audio_path)
                print(f"VIDEO_STATUS:{video_id}:Status: Deleted audio file: {wav_audio_path}") # Structured status
            except Exception as e:
                warning_msg = f"Failed to delete audio file {wav_audio_path}: {e}"
                logging.warning(warning_msg)
                print(f"VIDEO_ERROR:{video_id}:Error: {warning_msg}") # Structured error (using error for failed cleanup)


        # Correct transcript using selected models
        corrected_transcript = ""
        print(f"VIDEO_STATUS:{video_id}:Starting transcript correction") # Structured status
        for model in correction_models:
            try:
                status_msg = f"Attempting correction with model: {model}"
                logging.info(status_msg)
                print(f"VIDEO_STATUS:{video_id}:Status: {status_msg}") # Structured status
                # Pass the segments with speaker info to correct_transcript
                corrected_transcript = correct_transcript(
                    segments_with_speakers, correction_model=model # Use segments_with_speakers
                )
                corrected_path = os.path.join(
                    video_folder,
                    f"{sanitized_title}_corrected_transcript_{model.replace('/', '_')}.txt",
                )
                with open(corrected_path, "w", encoding="utf-8") as f:
                    f.write(corrected_transcript)
                logging.info(f"Corrected transcript saved: {corrected_path}")
                print(f"VIDEO_STATUS:{video_id}:Status: Corrected transcript saved to {corrected_path}") # Structured status
                video_stats["steps_completed"].append(f"correction_{model}")
            except Exception as e:
                error_msg = f"Transcript correction failed with model {model} for {video_title}: {e}"
                logging.error(error_msg, exc_info=True)
                print(f"VIDEO_ERROR:{video_id}:Error: {error_msg}") # Structured error
                video_stats["error"] = video_stats.get("error", "") + f"Correction ({model}) failed: {e}\n" # Append error

        # Summarize transcript
        summary = ""
        if corrected_transcript and summarization_model: # Only summarize if correction was successful and a model is specified
            print(f"VIDEO_STATUS:{video_id}:Starting transcript summarization") # Structured status
            try:
                status_msg = f"Attempting summarization with model: {summarization_model}"
                logging.info(status_msg)
                print(f"VIDEO_STATUS:{video_id}:Status: {status_msg}") # Structured status
                summary = summarize_transcript(corrected_transcript, summarization_model)
                summary_path = os.path.join(
                    video_folder,
                    f"{sanitized_title}_summary_{summarization_model.replace('/', '_')}.txt",
                )
                with open(summary_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                logging.info(f"Summary saved: {summary_path}")
                print(f"VIDEO_STATUS:{video_id}:Status: Summary saved to {summary_path}") # Structured status
                video_stats["steps_completed"].append("summarization")
            except Exception as e:
                error_msg = f"Transcript summarization failed with model {summarization_model} for {video_title}: {e}"
                logging.error(error_msg, exc_info=True)
                print(f"VIDEO_ERROR:{video_id}:Error: {error_msg}") # Structured error
                video_stats["error"] = video_stats.get("error", "") + f"Summarization ({summarization_model}) failed: {e}\n" # Append error
        elif not corrected_transcript:
             warning_msg = "Skipping summarization as transcript correction failed."
             logging.warning(warning_msg)
             print(f"VIDEO_STATUS:{video_id}:Status: {warning_msg}") # Structured status
        elif not summarization_model:
             warning_msg = "Skipping summarization as no summarization model was specified."
             logging.warning(warning_msg)
             print(f"VIDEO_STATUS:{video_id}:Status: {warning_msg}") # Structured status


        # Generate HTML reader
        if corrected_transcript: # Only generate HTML if correction was successful
            print(f"VIDEO_STATUS:{video_id}:Generating HTML reader") # Structured status
            html_path = os.path.join(video_folder, f"{sanitized_title}_transcript_reader.html")
            try:
                # Pass the corrected transcript to the HTML generator
                workflow_logic.generate_html_reader(corrected_transcript, html_path)
                logging.info(f"HTML reader generated: {html_path}")
                print(f"VIDEO_STATUS:{video_id}:Status: HTML reader generated at {html_path}") # Structured status
                video_stats["steps_completed"].append("html_reader")
            except Exception as e:
                error_msg = f"Failed to generate HTML reader for {video_title}: {e}"
                logging.error(error_msg, exc_info=True)
                print(f"VIDEO_ERROR:{video_id}:Error: {error_msg}") # Structured error
                video_stats["error"] = video_stats.get("error", "") + f"HTML reader generation failed: {e}\n" # Append error
        else:
            warning_msg = "Skipping HTML reader generation as transcript correction failed."
            logging.warning(warning_msg)
            print(f"VIDEO_STATUS:{video_id}:Status: {warning_msg}") # Structured status


        # Finalize video stats
        video_stats["end_time"] = datetime.datetime.now().isoformat()
        video_stats["duration_seconds"] = time.time() - start_time
        video_stats["success"] = video_stats.get("error") is None # Mark as success if no errors occurred

        stats_path = os.path.join(video_folder, f"{sanitized_title}_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(video_stats, f, indent=2)
        logging.info(f"Video stats saved: {stats_path}")
        print(f"VIDEO_STATUS:{video_id}:Status: Video stats saved to {stats_path}") # Structured status

        if video_stats["success"]:
            stats["processed_videos"] += 1
            print(f"VIDEO_STATUS:{video_id}:Completed processing for video: {video_title}") # Structured status
        else:
            stats["failed_videos"] += 1
            print(f"VIDEO_STATUS:{video_id}:Failed processing for video: {video_title}") # Structured status


        pbar.update(1) # Update progress bar

    pbar.close() # Close progress bar

    # Finalize overall stats
    stats["end_time"] = datetime.datetime.now().isoformat()
    logging.info("Pipeline execution finished.")
    print("PIPELINE_STATUS:Completed") # Structured status for pipeline completion

    # Save overall stats
    stats_summary_path = os.path.join(output_base_dir, "pipeline_stats_summary.json")
    with open(stats_summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Overall pipeline stats saved: {stats_summary_path}")
    print(f"PIPELINE_STATUS:Overall stats saved to {stats_summary_path}") # Structured status

    # Print summary of results
    print("\n==================== Pipeline Summary ====================")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Processed successfully: {stats['processed_videos']}")
    print(f"Skipped: {stats['skipped_videos']}")
    print(f"Failed: {stats['failed_videos']}")
    print("==========================================================")

    if stats["failed_videos"] > 0:
        logging.error(f"Pipeline finished with {stats['failed_videos']} failed videos.")
        # Optionally, list failed videos
        # for video in video_list:
        #     video_folder = os.path.join(output_base_dir, sanitize_filename(video.get("title", "")))
        #     stats_path = os.path.join(video_folder, f"{sanitize_filename(video.get('title', ''))}_stats.json")
        #     if os.path.exists(stats_path):
        #         with open(stats_path, 'r', encoding='utf-8') as f:
        #             video_stats = json.load(f)
        #         if not video_stats.get("success", True):
        #             print(f"Failed video: {video.get('title', 'Unknown Title')} ({video.get('id', 'Unknown ID')}) - Error: {video_stats.get('error', 'No error details')}")

    else:
        logging.info("Pipeline finished successfully with no failed videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the YouTube transcription and processing pipeline.")
    parser.add_argument(
        "playlist_url", help="The URL of the YouTube playlist to process."
    )
    parser.add_argument(
        "--output_base_dir",
        default="output",
        help="The base directory to save the output files.",
    )
    parser.add_argument(
        "--correction_models",
        nargs="+", # Allows multiple model names
        default=["google/gemini-2.5-flash-preview"], # Default correction model
        help="List of LLM models to use for transcript correction (via OpenRouter).",
    )
    parser.add_argument(
        "--summarization_model",
        default="google/gemini-2.5-flash-preview", # Default summarization model
        help="The LLM model to use for transcript summarization (via OpenRouter).",
    )
    parser.add_argument(
        "--video_ids",
        nargs="+", # Allows multiple video IDs
        default=None, # Default to None to process all videos
        help="Optional: List of specific video IDs to process from the playlist.",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            playlist_url=args.playlist_url,
            output_base_dir=args.output_base_dir,
            correction_models=args.correction_models,
            summarization_model=args.summarization_model,
            video_ids=args.video_ids, # Pass the video_ids argument
        )
    except Exception as e:
        logging.error(f"An unhandled error occurred during pipeline execution: {e}", exc_info=True)
        print(f"PIPELINE_ERROR:An unhandled error occurred: {e}") # Structured error

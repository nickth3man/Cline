import os
import json
import logging
import re
import yt_dlp
import datetime
import time
from tqdm import tqdm
from src.utils.workflow_logic import (
    transcribe_audio,
    diarize_speakers,
    correct_transcript,
)
from src.utils import workflow_logic
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
    except Exception as e:
        logging.error(f"Failed to fetch playlist info: {e}", exc_info=True)
        raise RuntimeError(f"Playlist fetching failed: {e}")

    # Create a progress bar for the overall process
    pbar = tqdm(total=len(video_list), desc="Processing videos", unit="video")

    # Process each video
    for index, video in enumerate(video_list):
        start_time = time.time()
        video_title = video.get("title", f"video_{index}")
        video_url = video.get("url") or video.get("webpage_url") or video.get("id")

        # Update progress bar description
        pbar.set_description(f"Processing: {video_title[:30]}...")

        # Skip if no valid URL is found (e.g., deleted video)
        if (
            not video_url
            or "unavailable" in video_title.lower()
            or "deleted" in video_title.lower()
        ):
            logging.warning(
                f"Skipping video {index+1}/{len(video_list)} ('{video_title}') due to missing URL or unavailable status."
            )
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
                logging.info(f"Downloading audio (attempt {attempt+1}/3): {video_url}")

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
                        logging.info(
                            f"Successfully downloaded audio on attempt {attempt+1}"
                        )
                        break
                    except Exception as e:
                        logging.warning(
                            f"Could not rename file {attempt_audio_path} to {wav_audio_path}: {e}"
                        )
                        # Use the attempt file directly if we can't rename
                        wav_audio_path = attempt_audio_path
                        audio_downloaded = True
                        logging.info(f"Using attempt file directly: {wav_audio_path}")
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
                    logging.warning(
                        f"Expected WAV not found, using alternative: {actual_audio_file}"
                    )
                    wav_audio_path = actual_audio_file
                    audio_downloaded = True
                    break
                else:
                    logging.warning(
                        f"No audio files found on attempt {attempt+1}, trying next method..."
                    )
                    # Give the file system time to release any locks before next attempt
                    time.sleep(2)

            except Exception as e:
                logging.error(
                    f"Audio download failed on attempt {attempt+1}: {e}", exc_info=True
                )
                if attempt < 2:  # Only log warning if we have more attempts left
                    logging.info(f"Retrying with different download method...")
                    # Give the file system time to release any locks before next attempt
                    time.sleep(2)
                else:
                    error_msg = f"All audio download attempts failed for {video_title}"
                    logging.error(error_msg)
                    video_stats["error"] = error_msg

        if not audio_downloaded:
            logging.error(
                f"Failed to download audio after multiple attempts for {video_title}"
            )
            video_stats["end_time"] = datetime.datetime.now().isoformat()
            video_stats["duration_seconds"] = time.time() - start_time
            stats_path = os.path.join(video_folder, f"{sanitized_title}_stats.json")
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(video_stats, f, indent=2)
            stats["failed_videos"] += 1
            pbar.update(1)
            continue  # Skip to next video

        logging.info(f"Downloaded and converted audio to {wav_audio_path}")
        video_stats["steps_completed"].append("audio_download")

        # Transcribe audio
        try:
            raw_transcript = transcribe_audio(wav_audio_path)
            raw_transcript_path = os.path.join(
                video_folder, f"{sanitized_title}_raw_transcript.txt"
            )
            with open(raw_transcript_path, "w", encoding="utf-8") as f:
                f.write(raw_transcript)
            logging.info(f"Raw transcript saved: {raw_transcript_path}")
            video_stats["steps_completed"].append("transcription")
        except Exception as e:
            error_msg = f"Transcription failed for {video_title}: {e}"
            logging.error(error_msg, exc_info=True)
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

        # Diarize speakers (Pseudo-diarization via LLM)
        diarization_result = []
        diarization_path = os.path.join(
            video_folder, f"{sanitized_title}_diarization.json"
        )
        try:
            diarization_result = diarize_speakers(wav_audio_path, raw_transcript)
            with open(diarization_path, "w", encoding="utf-8") as f:
                json.dump(diarization_result, f, indent=2)
            logging.info(f"Diarization results saved: {diarization_path}")
            video_stats["steps_completed"].append("diarization")
        except Exception as e:
            error_msg = f"Diarization failed for {video_title}: {e}"
            logging.error(error_msg, exc_info=True)
            video_stats["error"] = error_msg
            # Create a minimal diarization result to allow continuing
            diarization_result = [
                {"speaker": "Speaker", "start": 0, "end": 0, "text": ""}
            ]
            with open(diarization_path, "w", encoding="utf-8") as f:
                json.dump(diarization_result, f, indent=2)
            logging.warning(
                f"Created minimal diarization result to continue pipeline: {diarization_path}"
            )

        # Correct transcript using selected models
        corrected_transcript = ""
        for model in correction_models:
            try:
                corrected_transcript = correct_transcript(
                    raw_transcript, diarization_result, correction_model=model
                )
                corrected_path = os.path.join(
                    video_folder,
                    f"{sanitized_title}_corrected_transcript_{model.replace('/', '_')}.txt",
                )
                with open(corrected_path, "w", encoding="utf-8") as f:
                    f.write(corrected_transcript)
                logging.info(f"Corrected transcript ({model}) saved: {corrected_path}")
                video_stats["steps_completed"].append("correction")
                break  # Stop after first successful correction
            except Exception as e:
                error_msg = f"Transcript correction failed with model {model} for {video_title}: {e}"
                logging.error(error_msg, exc_info=True)
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
        if not corrected_transcript:
            logging.warning(
                f"Transcript correction failed for all models for {video_title}. Using raw transcript."
            )
            corrected_transcript = raw_transcript  # Fallback

        # Summarize transcript
        try:
            summary = workflow_logic.summarize_transcript(
                corrected_transcript, summarization_model
            )
            summary_path = os.path.join(
                video_folder,
                f"{sanitized_title}_summary_{summarization_model.replace('/', '_')}.txt",
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            logging.info(f"Summary saved: {summary_path}")
            video_stats["steps_completed"].append("summarization")
        except Exception as e:
            error_msg = f"Summarization failed for {video_title}: {e}"
            logging.error(error_msg, exc_info=True)
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

        # Generate HTML reader
        try:
            html_path = os.path.join(
                video_folder, f"{sanitized_title}_transcript_reader.html"
            )
            generate_html_reader(corrected_transcript, html_path)
            logging.info(f"HTML transcript reader saved: {html_path}")
            video_stats["steps_completed"].append("html_reader")
        except Exception as e:
            error_msg = f"HTML reader generation failed for {video_title}: {e}"
            logging.error(error_msg, exc_info=True)
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

        # Save metadata
        try:
            metadata = {
                "title": video_title,
                "url": video_url,
                "playlist_url": playlist_url,
                "processed_at": datetime.datetime.now().isoformat(),
                "correction_models_attempted": correction_models,
                "correction_model_used": (
                    model
                    if corrected_transcript != raw_transcript
                    else "None (Fallback to raw)"
                ),
                "summarization_model": summarization_model,
                "files": {
                    "audio": os.path.basename(wav_audio_path),
                    "raw_transcript": os.path.basename(raw_transcript_path),
                    "diarization": os.path.basename(diarization_path),
                    "corrected_transcript": (
                        os.path.basename(corrected_path)
                        if corrected_transcript != raw_transcript
                        else None
                    ),
                    "summary": (
                        os.path.basename(summary_path)
                        if "summary_path" in locals()
                        else None
                    ),
                    "html_reader": (
                        os.path.basename(html_path) if "html_path" in locals() else None
                    ),
                },
            }
            metadata_path = os.path.join(
                video_folder, f"{sanitized_title}_metadata.json"
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logging.info(f"Metadata saved: {metadata_path}")
            video_stats["steps_completed"].append("metadata")
        except Exception as e:
            error_msg = f"Metadata saving failed for {video_title}: {e}"
            logging.error(error_msg, exc_info=True)
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

        # Save video stats
        video_stats["success"] = True
        video_stats["end_time"] = datetime.datetime.now().isoformat()
        video_stats["duration_seconds"] = time.time() - start_time
        stats_path = os.path.join(video_folder, f"{sanitized_title}_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(video_stats, f, indent=2)

        # Update overall stats
        stats["processed_videos"] += 1
        stats["total_duration_seconds"] += video_stats["duration_seconds"]

        # Update progress bar
        pbar.update(1)

    # Close progress bar
    pbar.close()

    # Save overall stats
    stats["end_time"] = datetime.datetime.now().isoformat()
    stats_path = os.path.join(output_base_dir, "pipeline_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logging.info(
        f"Pipeline finished. Processed {stats['processed_videos']}/{stats['total_videos']} videos in {stats['total_duration_seconds']:.2f} seconds."
    )
    logging.info(
        f"Skipped: {stats['skipped_videos']}, Failed: {stats['failed_videos']}"
    )
    logging.info(f"Overall stats saved to: {stats_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python run_full_pipeline.py <playlist_url>")
        sys.exit(1)
    playlist_url = sys.argv[1]
    output_dir = "output"
    correction_models = ["openai/gpt-4.1-mini", "anthropic/claude-3-haiku-20240307"]
    summarization_model = "openai/gpt-4.1-mini"
    run_pipeline(playlist_url, output_dir, correction_models, summarization_model)

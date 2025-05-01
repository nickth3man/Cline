"""
Core Workflow Logic for Audio Processing

This module contains the core functions for processing audio files, including
transcription, speaker diarization, and LLM-based transcript correction.
"""

import os
import subprocess
import logging
import datetime
import traceback
import json
import shutil
import platform
import psutil
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
import math
import io

from openai import OpenAI
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.utils import mediainfo

import whisperx
import gc # Import gc for potential model flushing
import pandas as pd # Import pandas for DataFrame check

# --- Load Environment Variables ---
# It's generally better practice to load this once at the application entry point,
# but loading here ensures the module is self-contained if used independently.
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for pyannote models

# --- Initialize OpenAI Client for OpenRouter ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def get_system_info() -> Dict[str, Any]:
    """
    Collect system information for diagnostics.

    Returns:
        Dict containing system information categories
    """
    # OS information
    system_info = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
            "path": sys.path,
        },
        "hardware": {
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_memory_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            ),
            "memory_percent_used": psutil.virtual_memory().percent,
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
        },
        "gpu": {
            "cuda_available": torch.cuda.is_available() if "torch" in sys.modules else False,
        },
        "environment": {
            "PATH_available": "PATH" in os.environ,
            "PYTHONPATH_available": "PYTHONPATH" in os.environ,
            "VIRTUAL_ENV_available": "VIRTUAL_ENV" in os.environ,
            "VIRTUAL_ENV": os.getenv("VIRTUAL_ENV"),
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }
    return system_info


# Configure logging with timestamps and levels
# This is also configured in run_full_pipeline.py, but keeping it here
# for standalone usage or debugging of this module.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        # Using a relative path that assumes the log directory is at the project root
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "..", "..", "logs", "workflow_logic.log")
        ),
    ],
)

# Log system information at startup
system_info = get_system_info()
logging.info("Starting workflow_logic module. Log file: logs/workflow_logic.log")
logging.info(f"System Information:\n{json.dumps(system_info, indent=2)}")


def check_ffmpeg():
    """Checks if FFmpeg is installed and accessible in the system's PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info("FFmpeg found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error(
            "FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH."
        )
        # Provide instructions based on OS
        if platform.system() == "Windows":
            logging.info(
                "For Windows, you can download FFmpeg from https://ffmpeg.org/download.html and add its bin directory to your system's PATH environment variable."
            )
        elif platform.system() == "Darwin":  # macOS
            logging.info("For macOS, you can install FFmpeg using Homebrew: brew install ffmpeg")
        else:  # Linux and others
            logging.info(
                "For Linux, you can usually install FFmpeg using your distribution's package manager (e.g., sudo apt-get install ffmpeg or sudo yum install ffmpeg)."
            )
        raise EnvironmentError("FFmpeg is required but not found.")


# Check for FFmpeg at module load time
try:
    check_ffmpeg()
except EnvironmentError:
    # Decide how to handle this error: exit, disable features, etc.
    # For now, we'll let the exception propagate if FFmpeg is critical.
    # If FFmpeg is only needed for specific functions, handle the error there.
    pass  # Allow the module to load even if FFmpeg is missing, but log the error.


def initialize_diarization_pipeline(model_name: str = "pyannote/speaker-diarization-3.1"):
    """
    Initializes the pyannote.audio diarization pipeline.

    Args:
        model_name: The name of the pyannote.audio model to use.

    Returns:
        The initialized pyannote.audio Pipeline object.

    Raises:
        EnvironmentError: If the Hugging Face token (HF_TOKEN) is not set.
        Exception: If there's an error loading the pipeline.
    """
    if not HF_TOKEN:
        raise EnvironmentError(
            "Hugging Face token (HF_TOKEN) is not set. Please set it in your .env file or environment variables."
        )

    logging.info(f"Initializing pyannote.audio pipeline with model: {model_name}")
    try:
        # Use the HF_TOKEN for authentication
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        logging.info("pyannote.audio pipeline initialized successfully.")
        return pipeline
    except Exception as e:
        logging.error(f"Error initializing pyannote.audio pipeline: {e}", exc_info=True)
        raise Exception(f"Failed to initialize pyannote.audio pipeline: {e}")


# Initialize diarization pipeline at module load time
# This assumes the diarization pipeline is needed for all uses of this module.
# If it's only needed for specific functions, initialize it within those functions.
try:
    diarization_pipeline = initialize_diarization_pipeline()
except (EnvironmentError, Exception) as e:
    logging.error(f"Diarization pipeline initialization failed: {e}")
    diarization_pipeline = None  # Set to None if initialization fails

# --- Initialize WhisperX Models ---
# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 # Reduce if you run out of memory
compute_type = "float16" if torch.cuda.is_available() else "int8" # Use float16 for GPU, int8 for CPU

logging.info(f"Initializing WhisperX models on device: {device} with compute type: {compute_type}")

try:
    # Load WhisperX model
    whisperx_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    logging.info("WhisperX transcription model loaded successfully.")

    # Load alignment model and metadata
    whisperx_align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    logging.info("WhisperX alignment model loaded successfully (language: en).")

    # Initialize diarization model (using pyannote via whisperx)
    # This requires the HF_TOKEN and pyannote dependencies
    if HF_TOKEN and diarization_pipeline: # Check if pyannote pipeline initialized successfully
         whisperx_diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
         logging.info("WhisperX diarization model (using pyannote) initialized successfully.")
    else:
         logging.warning("Hugging Face token not available or pyannote pipeline failed to initialize. WhisperX diarization will not be available.")
         whisperx_diarize_model = None


except Exception as e:
    logging.error(f"Error initializing WhisperX models: {e}", exc_info=True)
    whisperx_model = None
    whisperx_align_model = None
    metadata = None
    whisperx_diarize_model = None
    logging.error("WhisperX initialization failed. Transcription will not be available.")


def transcribe_audio(video_id: str, audio_path: str) -> Dict[str, Any]:
    """
    Transcribes the given audio file using a self-hosted WhisperX model and performs Pyannote diarization.

    Args:
        video_id: The ID of the video being processed.
        audio_path: Path to the audio file.

    Returns:
        A dictionary containing the transcription result, including segments with timestamps, text, and speaker assignments.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If WhisperX models are not loaded or transcription/diarization fails.
        EnvironmentError: If the diarization pipeline is not initialized.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")

    # Check if WhisperX models were loaded successfully
    if whisperx_model is None or whisperx_align_model is None or metadata is None:
        error_message = "WhisperX models are not loaded. Cannot perform transcription."
        logging.error(f"VIDEO_ERROR:{video_id}:{error_message}")
        raise RuntimeError(error_message)

    # Check if diarization pipeline was loaded successfully
    if diarization_pipeline is None:
         error_message = "Pyannote.audio diarization pipeline is not loaded. Cannot perform diarization."
         logging.error(f"VIDEO_ERROR:{video_id}:{error_message}")
         raise EnvironmentError(error_message)

    logging.info(f"VIDEO_STATUS:{video_id}:Starting transcription and diarization...")

    try:
        # Load audio
        audio = whisperx.load_audio(audio_path)

        # Transcribe audio
        logging.info(f"VIDEO_STATUS:{video_id}:Running WhisperX transcription...")
        # Added more specific error handling around the transcribe call
        try:
            result = whisperx_model.transcribe(audio, batch_size=batch_size)
            logging.info(f"VIDEO_STATUS:{video_id}:WhisperX transcription complete.")
        except Exception as transcribe_error:
            error_message = f"WhisperX transcription failed for {video_id}: {transcribe_error}"
            logging.error(f"VIDEO_ERROR:{video_id}:{error_message}", exc_info=True)
            raise RuntimeError(error_message) from transcribe_error


        # Align transcription
        logging.info(f"VIDEO_STATUS:{video_id}:Running WhisperX alignment...")
        try:
            result = whisperx.align(result["segments"], whisperx_align_model, metadata, audio, device, return_char_alignments=False)
            logging.info(f"VIDEO_STATUS:{video_id}:WhisperX alignment complete.")
        except Exception as align_error:
            error_message = f"WhisperX alignment failed for {video_id}: {align_error}"
            logging.error(f"VIDEO_ERROR:{video_id}:{error_message}", exc_info=True)
            raise RuntimeError(error_message) from align_error

        # Perform diarization using pyannote.audio pipeline
        logging.info(f"VIDEO_STATUS:{video_id}:Running Pyannote.audio diarization...")
        try:
            diarization_result = diarization_pipeline(audio_path)
            logging.info(f"VIDEO_STATUS:{video_id}:Pyannote.audio diarization complete.")
        except Exception as diarization_error:
            error_message = f"Pyannote.audio diarization failed for {video_id}: {diarization_error}"
            logging.error(f"VIDEO_ERROR:{video_id}:{error_message}", exc_info=True)
            raise RuntimeError(error_message) from diarization_error


        # Assign speakers to segments
        logging.info(f"VIDEO_STATUS:{video_id}:Assigning speakers to segments...")
        segments_with_speakers = assign_speakers_to_segments(result["segments"], diarization_result)
        logging.info(f"VIDEO_STATUS:{video_id}:Speaker assignment complete.")

        # The segments_with_speakers now contains the transcribed segments with timestamps and speaker labels.
        formatted_segments = []
        full_text = ""
        for segment in segments_with_speakers:
            formatted_segments.append({
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", "").strip(),
                "speaker": segment.get("speaker", "Unknown Speaker"), # Include speaker information
            })
            full_text += f"[{segment.get('speaker', 'Unknown Speaker')}] {segment.get('text', '').strip()} " # Concatenate text with speaker for full transcript

        # Get total duration (optional, but good for compatibility)
        try:
            audio_info = mediainfo(audio_path)
            total_duration_seconds = float(audio_info.get('duration', 0))
        except Exception as e:
            logging.warning(f"VIDEO_STATUS:{video_id}:Could not get audio duration using mediainfo: {e}")
            total_duration_seconds = None # Or try to estimate from last segment end time

        logging.info(f"VIDEO_STATUS:{video_id}:Transcription, alignment, and diarization successful.")

        # Return the combined segments in a dictionary format compatible with subsequent steps
        return {
            "segments": formatted_segments,
            "text": full_text.strip(),
            "language": result.get("language"), # Include language if WhisperX detected it
            "duration": total_duration_seconds
        }

    except Exception as e:
        # This catch block will now primarily catch errors from assign_speakers_to_segments
        # or any unexpected errors not caught by the more specific blocks above.
        error_message = f"Transcription/Diarization post-processing failed for {video_id}: {e}"
        logging.error(f"VIDEO_ERROR:{video_id}:{error_message}", exc_info=True)
        raise RuntimeError(error_message) from e
    finally:
        # Optional: Clear GPU memory after transcription if needed
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


def assign_speakers_to_segments(segments: List[Dict[str, Any]], diarization_result) -> List[Dict[str, Any]]:
    """
    Assigns speaker labels from the diarization result to the corresponding transcription segments.

    Args:
        segments: A list of transcription segments with start and end timestamps.
        diarization_result: The diarization result from pyannote.audio (an Annotation object).

    Returns:
        A list of transcription segments with added 'speaker' keys.
    """
    segments_with_speakers = []
    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        # Find the speaker who speaks the most within the segment's time frame
        speaker_votes = {}
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            turn_start = turn.start
            turn_end = turn.end

            # Calculate overlap between segment and speaker turn
            overlap_start = max(segment_start, turn_start)
            overlap_end = min(segment_end, turn_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                speaker_votes[speaker] = speaker_votes.get(speaker, 0) + overlap_duration

        # Assign the speaker with the maximum overlap duration
        if speaker_votes:
            assigned_speaker = max(speaker_votes, key=speaker_votes.get)
            segment["speaker"] = assigned_speaker
        else:
            segment["speaker"] = "Unknown Speaker" # Default if no speaker turn overlaps

        segments_with_speakers.append(segment)

    return segments_with_speakers


def correct_transcript(
    transcription_segments: List[Dict[str, Any]],
    correction_model: str = "google/gemini-2.5-flash-preview", # Set default model
) -> str:
    """
    Corrects and formats the transcript using an LLM, incorporating speaker information.

    Args:
        transcription_segments: A list of transcribed segments with speaker assignments.
        correction_model: The LLM model to use for correction via OpenRouter.

    Returns:
        The corrected and formatted transcript.

    Raises:
        ConnectionError: If the OpenAI client is not initialized.
        RuntimeError: If the LLM API call fails or returns empty content.
    """
    logging.info(
        f"Starting transcript correction using model '{correction_model}' via OpenRouter..."
    )

    # Format the input for the LLM using the segments with speaker information
    formatted_input = "Transcript Segments (with speaker and timestamps):\n"
    for segment in transcription_segments:
        start_time = segment.get("start", "N/A")
        end_time = segment.get("end", "N/A")
        speaker = segment.get("speaker", "Unknown Speaker")
        text = segment.get("text", "")
        # Ensure timestamps are formatted correctly if they are numbers
        try:
            start_time_formatted = f"{float(start_time):.2f}" if isinstance(start_time, (int, float)) else str(start_time)
            end_time_formatted = f"{float(end_time):.2f}" if isinstance(end_time, (int, float)) else str(end_time)
            formatted_input += f"[{start_time_formatted}-{end_time_formatted}] {speaker}: {text}\n"
        except (ValueError, TypeError):
             formatted_input += f"[{start_time}-{end_time}] {speaker}: {text}\n" # Fallback if timestamps aren't numeric


    try:
        # Use the chat completions endpoint for text correction
        response = client.chat.completions.create(
            model=correction_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert transcription corrector. Correct the provided transcript segments for grammar, punctuation, and spelling. Do not add or remove information. Maintain the original meaning. Preserve the speaker labels and timestamps in the output.",
                },
                {"role": "user", "content": formatted_input},
            ],
            temperature=0.1,  # Keep corrections focused and not creative
        )
        corrected_transcript = response.choices[0].message.content.strip()
        if not corrected_transcript:
            warning_message = "Transcript correction API call succeeded but returned empty content."
            logging.warning(warning_message)
            # Raise an error to indicate failure due to empty content
            raise RuntimeError(warning_message)

        logging.info("Transcript correction successful.")
        return corrected_transcript
    except Exception as e:
        error_message = f"Transcript correction API call failed with model {correction_model}: {e}"
        logging.error(error_message, exc_info=True)
        raise RuntimeError(error_message) from e


def summarize_transcript(transcript: str, summarization_model: str) -> str:
    """
    Summarizes the given transcript using an LLM.

    Args:
        transcript: The transcript text (preferably corrected).
        summarization_model: The LLM model to use for summarization via OpenRouter.

    Returns:
        The summarized text.

    Raises:
        ConnectionError: If the OpenAI client is not initialized.
        RuntimeError: If the LLM API call fails or returns empty content.
    """
    if not client:
        raise ConnectionError(
            "OpenAI client not initialized. Check OPENROUTER_API_KEY."
        )

    logging.info(
        f"Starting transcript summarization using model '{summarization_model}' via OpenRouter..."
    )

    try:
        # Use the chat completions endpoint for summarization
        response = client.chat.completions.create(
            model=summarization_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert summarizer. Summarize the provided transcript concisely and accurately. Focus on the main points and key information.",
                },
                {"role": "user", "content": transcript},
            ],
            temperature=0.3,  # Allow some creativity for summarization
        )
        summary = response.choices[0].message.content.strip()
        if not summary:
            warning_message = "Transcript summarization API call succeeded but returned empty content."
            logging.warning(warning_message)
            # Raise an error to indicate failure due to empty content
            raise RuntimeError(warning_message)

        logging.info("Transcript summarization successful.")
        return summary
    except Exception as e:
        error_message = f"Transcript summarization API call failed with model {summarization_model}: {e}"
        logging.error(error_message, exc_info=True)
        raise RuntimeError(error_message) from e


def generate_html_reader(corrected_transcript: str, output_path: str):
    """
    Generates a simple HTML file to display the corrected transcript.

    Args:
        corrected_transcript: The corrected transcript text.
        output_path: The path where the HTML file will be saved.
    """
    logging.info(f"Generating HTML reader at {output_path}...")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcript Reader</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; margin: 20px; }}
        .container {{ max-width: 800px; margin: auto; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Transcript</h1>
        <pre>{corrected_transcript}</pre>
    </div>
</body>
</html>
"""

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML reader saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error generating or saving HTML reader to {output_path}: {e}", exc_info=True)
        # Decide if this should raise an error or just log a warning
        # For now, logging a warning seems sufficient as it's a presentation layer issue
        pass


def save_transcript_to_file(transcript_data: Union[str, List[Dict[str, Any]]], output_path: str, file_format: str = "txt"):
    """
    Saves transcript data to a file in the specified format.

    Args:
        transcript_data: The transcript data (either raw text or list of segments).
        output_path: The path where the file will be saved.
        file_format: The desired file format ('txt' or 'json').
    """
    logging.info(f"Saving transcript data to {output_path} in format {file_format}...")

    try:
        if file_format == "txt":
            if isinstance(transcript_data, str):
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcript_data)
            elif isinstance(transcript_data, list):
                # If segments are provided for TXT, format them with speaker/timestamps
                formatted_text = ""
                for segment in transcript_data:
                    start_time = segment.get("start", "N/A")
                    end_time = segment.get("end", "N/A")
                    speaker = segment.get("speaker", "Unknown Speaker")
                    text = segment.get("text", "")
                    try:
                        start_time_formatted = f"{float(start_time):.2f}" if isinstance(start_time, (int, float)) else str(start_time)
                        end_time_formatted = f"{float(end_time):.2f}" if isinstance(end_time, (int, float)) else str(end_time)
                        formatted_text += f"[{start_time_formatted}-{end_time_formatted}] {speaker}: {text}\n"
                    except (ValueError, TypeError):
                        formatted_text += f"[{start_time}-{end_time}] {speaker}: {text}\n" # Fallback if timestamps aren't numeric

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(formatted_text)
            else:
                raise TypeError(f"Unsupported data type for txt format: {type(transcript_data)}")

        elif file_format == "json":
            if isinstance(transcript_data, list):
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(transcript_data, f, indent=2)
            else:
                 # If raw text is provided for JSON, wrap it in a simple structure
                 logging.warning(f"Saving raw text as JSON for {output_path}. Wrapping in 'text' key.")
                 with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"text": transcript_data}, f, indent=2)

        else:
            raise ValueError(f"Unsupported file format: {file_format}. Supported formats are 'txt' and 'json'.")

        logging.info(f"Transcript data saved successfully to {output_path}")

    except Exception as e:
        logging.error(f"Error saving transcript data to {output_path}: {e}", exc_info=True)

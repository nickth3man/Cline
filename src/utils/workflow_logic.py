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

from openai import OpenAI
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv

# --- Load Environment Variables ---
# It's generally better practice to load this once at the application entry point,
# but loading here ensures the module is self-contained if used independently.
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for pyannote models


def get_system_info() -> Dict[str, Any]:
    """
    Collect system information for diagnostics.

    Returns:
        Dict containing system information categories
    """
    # OS information
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # Python information
    python_info = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
        "path": str(sys.path),
    }

    # Hardware information
    try:
        memory_info = {
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_memory_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            ),
            "memory_percent_used": psutil.virtual_memory().percent,
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
        }
    except Exception as e:
        memory_info = {"error": str(e)}

    # GPU information via PyTorch
    gpu_info = {}
    try:
        gpu_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["current_device"] = torch.cuda.current_device()
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        gpu_info["error"] = str(e)

    # Environment variables (filtered for security)
    env_info = {
        "PATH_available": "PATH" in os.environ,
        "PYTHONPATH_available": "PYTHONPATH" in os.environ,
        "VIRTUAL_ENV_available": "VIRTUAL_ENV" in os.environ,
    }
    if "VIRTUAL_ENV" in os.environ:
        env_info["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]

    # Put it all together
    return {
        "os": os_info,
        "python": python_info,
        "hardware": memory_info,
        "gpu": gpu_info,
        "environment": env_info,
        "timestamp": datetime.datetime.now().isoformat(),
    }


# --- Configure Logging ---
# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up file handler for detailed logging
log_filename = f"logs/pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"
)
file_handler.setFormatter(file_format)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Remove any existing handlers to avoid duplicates
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

root_logger.addHandler(file_handler)

# Also keep console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_format)
root_logger.addHandler(console_handler)

# Log start of module and system information
logging.info(f"Starting workflow_logic module. Log file: {log_filename}")

# Log system information at startup
system_info = get_system_info()
logging.info("System Information:")
for key, value in system_info.items():
    if isinstance(value, dict):
        logging.info(f"  {key}:")
        for sub_key, sub_value in value.items():
            logging.info(f"    {sub_key}: {sub_value}")
    else:
        logging.info(f"  {key}: {value}")


# --- Error Classification ---
class PipelineError(Exception):
    """Base class for pipeline-specific errors."""

    def __init__(
        self, message: str, step: str, details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.step = step
        self.details = details or {}
        super().__init__(f"[{step}] {message}")


class TranscriptionError(PipelineError):
    """Error during audio transcription."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "transcription", details)


class DiarizationError(PipelineError):
    """Error during speaker diarization."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "diarization", details)


class CorrectionError(PipelineError):
    """Error during transcript correction."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "correction", details)


class SummarizationError(PipelineError):
    """Error during transcript summarization."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "summarization", details)


class FileOperationError(PipelineError):
    """Error during file operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "file_operation", details)


class CleanupError(PipelineError):
    """Error during cleanup operations."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "cleanup", details)


# --- System Information Utilities ---
def diagnose_error(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes an error and provides diagnostic information and potential solutions.

    Args:
        error: The exception that occurred
        context: Dictionary with contextual information about the operation

    Returns:
        Dictionary with diagnostic information and potential solutions
    """
    diagnosis = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "possible_causes": [],
        "suggested_solutions": [],
        "system_info": get_system_info(),
        "context": context,
    }

    # Analyze based on error type
    if isinstance(error, FileNotFoundError):
        diagnosis["possible_causes"].append("File does not exist or path is incorrect")
        diagnosis["suggested_solutions"].append(
            "Check if the file exists at the specified path"
        )
        diagnosis["suggested_solutions"].append("Verify file permissions")

    elif isinstance(error, PermissionError):
        diagnosis["possible_causes"].append(
            "Insufficient permissions to access or modify file"
        )
        diagnosis["suggested_solutions"].append(
            "Run the application with appropriate permissions"
        )
        diagnosis["suggested_solutions"].append("Check file and directory permissions")

    elif "OpenAI" in str(type(error)) or "API" in str(error):
        diagnosis["possible_causes"].append("API authentication or rate limit issue")
        diagnosis["possible_causes"].append("Network connectivity problem")
        diagnosis["suggested_solutions"].append(
            "Check API key and environment variables"
        )
        diagnosis["suggested_solutions"].append("Verify network connectivity")
        diagnosis["suggested_solutions"].append("Check for API service status")

    elif "CUDA" in str(error) or "GPU" in str(error):
        diagnosis["possible_causes"].append("GPU memory issue or driver problem")
        diagnosis["suggested_solutions"].append("Reduce batch size or model complexity")
        diagnosis["suggested_solutions"].append("Update GPU drivers")
        diagnosis["suggested_solutions"].append("Try running with CPU only")

    elif "memory" in str(error).lower():
        diagnosis["possible_causes"].append("Insufficient system memory")
        diagnosis["suggested_solutions"].append(
            "Close other applications to free memory"
        )
        diagnosis["suggested_solutions"].append("Process smaller audio files")
        diagnosis["suggested_solutions"].append("Upgrade system memory")

    # Add context-specific analysis
    if context.get("step") == "transcription":
        diagnosis["possible_causes"].append(
            "Audio file may be corrupted or in an unsupported format"
        )
        diagnosis["suggested_solutions"].append("Check audio file integrity")
        diagnosis["suggested_solutions"].append(
            "Convert audio to a standard format (16kHz WAV)"
        )

    elif context.get("step") == "diarization":
        diagnosis["possible_causes"].append(
            "Diarization model may not be properly loaded"
        )
        diagnosis["possible_causes"].append(
            "Audio quality issues affecting speaker detection"
        )
        diagnosis["suggested_solutions"].append("Verify HF_TOKEN environment variable")
        diagnosis["suggested_solutions"].append(
            "Check if pyannote.audio is properly installed"
        )

    # Log the diagnosis information
    logging.error(f"Error diagnosis for {context.get('step', 'unknown')} step:")
    logging.error(f"  Error type: {diagnosis['error_type']}")
    logging.error(f"  Error message: {diagnosis['error_message']}")
    logging.error(f"  Context: {context}")
    logging.error("  Possible causes:")
    for cause in diagnosis["possible_causes"]:
        logging.error(f"    - {cause}")
    logging.error("  Suggested solutions:")
    for solution in diagnosis["suggested_solutions"]:
        logging.error(f"    - {solution}")
    logging.error(f"  Traceback: \n{diagnosis['traceback']}")

    return diagnosis


# --- Error Recovery Utilities ---
def create_error_recovery_file(error_info: Dict[str, Any], output_dir: str) -> str:
    """
    Creates a recovery file with information about the error and how to resume processing.

    Args:
        error_info: Dictionary with error details
        output_dir: Directory to save the recovery file

    Returns:
        Path to the recovery file
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        recovery_file = os.path.join(
            output_dir,
            f"recovery_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        recovery_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "error_info": error_info,
            "recovery_instructions": {
                "manual_steps": [
                    "Check the error diagnosis for potential solutions",
                    "Fix the identified issues",
                    "Rerun the pipeline with the same parameters",
                ],
                "automatic_recovery": "Not yet implemented",  # Future feature
            },
        }

        with open(recovery_file, "w", encoding="utf-8") as f:
            json.dump(recovery_data, f, indent=2, ensure_ascii=False)

        logging.info(f"Created error recovery file: {recovery_file}")

        # Log recovery instructions
        logging.info("Recovery instructions:")
        for step in recovery_data["recovery_instructions"]["manual_steps"]:
            logging.info(f"  - {step}")

        return recovery_file

    except Exception as e:
        logging.error(f"Failed to create error recovery file: {str(e)}", exc_info=True)
        return ""


def backup_intermediate_files(audio_path: str, output_dir: str) -> Dict[str, str]:
    """
    Creates backups of intermediate files before cleanup for potential recovery.

    Args:
        audio_path: Path to the main audio file
        output_dir: Directory where output files are stored

    Returns:
        Dictionary mapping original paths to backup paths
    """
    backup_dir = os.path.join(
        "logs", "backups", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(backup_dir, exist_ok=True)

    backup_map = {}
    try:
        # Backup the main audio file
        if os.path.exists(audio_path):
            backup_path = os.path.join(backup_dir, os.path.basename(audio_path))
            shutil.copy2(audio_path, backup_path)
            backup_map[audio_path] = backup_path
            logging.info(f"Backed up main audio file: {audio_path} -> {backup_path}")

        # Backup intermediate audio files
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        for fname in os.listdir(base_dir):
            if (
                fname.startswith(base_name)
                and "audio_attempt" in fname
                and fname.endswith(".wav")
            ):
                file_path = os.path.join(base_dir, fname)
                backup_path = os.path.join(backup_dir, fname)
                shutil.copy2(file_path, backup_path)
                backup_map[file_path] = backup_path
                logging.info(
                    f"Backed up intermediate file: {file_path} -> {backup_path}"
                )

        logging.info(f"Created backups of {len(backup_map)} files in {backup_dir}")

        # Log detailed backup information
        for original, backup in backup_map.items():
            file_size = os.path.getsize(backup) / (1024 * 1024)  # Size in MB
            logging.debug(
                f"Backup details - File: {os.path.basename(original)}, Size: {file_size:.2f} MB, Path: {backup}"
            )

        return backup_map

    except Exception as e:
        logging.error(f"Failed to create backups: {str(e)}", exc_info=True)
        return backup_map


# --- Helper Functions ---


def check_ffmpeg() -> bool:
    """
    Checks if ffmpeg is accessible in the system PATH.

    Returns:
        True if ffmpeg is found, False otherwise.
    """
    try:
        # Use shell=True on Windows for better path searching, False otherwise
        use_shell = os.name == "nt"
        subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True, shell=use_shell
        )
        logging.info("FFmpeg found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning(
            "FFmpeg not found or not executable in system PATH. Audio extraction/conversion might fail."
        )
        return False
    except Exception as e:
        logging.error(f"Unexpected error checking for FFmpeg: {e}", exc_info=True)
        return False


# Cache the result of check_ffmpeg to avoid repeated checks
_ffmpeg_present = check_ffmpeg()


def extract_or_convert_audio(input_path: str, output_wav_path: str) -> str:
    """
    Extracts audio from video or converts an existing audio file to the required
    WAV format (16kHz, 16-bit PCM, mono) using FFmpeg.

    Args:
        input_path: Path to the input video or audio file.
        output_wav_path: Path where the output WAV file should be saved.

    Returns:
        The path to the generated WAV file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If FFmpeg is not found or if the conversion process fails.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not _ffmpeg_present:
        raise RuntimeError(
            "FFmpeg is required for audio preparation but was not found."
        )

    logging.info(
        f"Converting/Extracting audio to 16kHz mono WAV: '{input_path}' -> '{output_wav_path}'"
    )
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-vn",  # No video output
        "-acodec",
        "pcm_s16le",  # Standard WAV codec (signed 16-bit little-endian PCM)
        "-ac",
        "1",  # Mono channel
        "-ar",
        "16000",  # 16kHz sample rate (common requirement for ASR)
        "-nostdin",  # Prevent interference with stdin (good practice)
        "-y",  # Overwrite output file if it exists
        output_wav_path,
    ]
    try:
        # Use shell=True on Windows for better path searching, False otherwise
        use_shell = os.name == "nt"
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            shell=use_shell,
        )
        logging.info(f"FFmpeg conversion successful: {output_wav_path}")
        logging.debug(
            f"FFmpeg stderr: {process.stderr}"
        )  # Log stderr for debugging potential issues
        return output_wav_path
    except subprocess.CalledProcessError as e:
        error_message = f"FFmpeg failed for '{input_path}'. Error: {e.stderr}"
        logging.error(error_message)
        # Attempt to remove potentially corrupted output file
        if os.path.exists(output_wav_path):
            try:
                os.remove(output_wav_path)
            except OSError:
                pass
        raise RuntimeError(error_message) from e
    except FileNotFoundError:
        # This case should be caught by check_ffmpeg(), but as a fallback:
        logging.error("FFmpeg command not found during execution.")
        raise RuntimeError(
            "FFmpeg command not found. Please ensure FFmpeg is installed and in your PATH."
        )
    except Exception as e:
        # Catch unexpected errors during subprocess execution
        error_message = f"An unexpected error occurred during FFmpeg execution for '{input_path}': {str(e)}"
        logging.error(error_message, exc_info=True)
        raise RuntimeError(error_message) from e


# --- Core Workflow Functions ---


def transcribe_audio(
    audio_path: str, transcription_model: str = "openai/whisper-large-v3"
) -> str:
    """
    Transcribes the given audio file using the specified model via OpenRouter API.

    Args:
        audio_path: Path to the audio file (WAV format recommended).
        transcription_model: The identifier of the Whisper model to use on OpenRouter.

    Returns:
        The transcribed text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ConnectionError: If the OpenAI client is not initialized (missing API key).
        RuntimeError: If the transcription API call fails.
    """
    if not client:
        raise ConnectionError(
            "OpenAI client not initialized. Check OPENROUTER_API_KEY."
        )
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found for transcription: {audio_path}")

    logging.info(
        f"Starting transcription for '{audio_path}' using model '{transcription_model}' via OpenRouter..."
    )
    try:
        with open(audio_path, "rb") as audio_file:
            # Use a reasonable timeout for potentially large files
            transcript = client.with_options(timeout=600.0).audio.transcriptions.create(
                model=transcription_model,
                file=audio_file,
                # Add other parameters like 'language' or 'prompt' if needed
            )
        logging.info(f"Transcription successful for '{audio_path}'.")
        if transcript and transcript.text:
            logging.debug(f"Transcription text length: {len(transcript.text)}")
            return transcript.text
        else:
            logging.warning(f"Transcription returned empty text for '{audio_path}'.")
            return ""
    except Exception as e:
        logging.error(
            f"OpenRouter transcription API error for '{audio_path}': {e}", exc_info=True
        )
        # Re-raise a more specific error if possible, or the original one
        raise RuntimeError(f"Transcription failed: {e}") from e


def diarize_speakers(audio_path: str) -> List[Dict[str, Any]]:
    """
    Performs speaker diarization using the local pyannote.audio pipeline.

    Requires pyannote.audio and its models to be set up. HF_TOKEN might be needed.

    Args:
        audio_path: Path to the audio file (WAV format recommended).

    Returns:
        A list of speaker segments, each a dictionary:
        `{"speaker": label, "start": float_seconds, "end": float_seconds}`.
        Returns an empty list if the pipeline failed to load or if diarization fails.
    """
    global diarization_pipeline  # Access the globally loaded pipeline
    if not diarization_pipeline:
        logging.warning(
            f"Diarization pipeline not available. Skipping diarization for '{audio_path}'."
        )
        return []
    if not os.path.exists(audio_path):
        # Raise error instead of returning empty list, as this is unexpected if called correctly
        raise FileNotFoundError(f"Audio file not found for diarization: {audio_path}")

    logging.info(f"Starting speaker diarization for '{audio_path}'...")
    try:
        # Apply pipeline with default hyperparameters
        # You might need to adjust `min_duration_on`, `min_duration_off` for specific audio
        # num_speakers=None lets the pipeline detect the number of speakers
        diarization = diarization_pipeline(audio_path, num_speakers=None)

        segments = []
        # Convert pyannote annotation to a simpler list format with float seconds
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": speaker,  # e.g., SPEAKER_00, SPEAKER_01
                    "start": round(turn.start, 3),  # Round to milliseconds
                    "end": round(turn.end, 3),
                }
            )

        # Sort segments by start time (pyannote usually guarantees it, but good practice)
        segments.sort(key=lambda x: x["start"])
        num_speakers_found = len(diarization.labels())
        logging.info(
            f"Diarization successful for '{audio_path}'. Found {len(segments)} segments for {num_speakers_found} speaker(s)."
        )
        return segments
    except Exception as e:
        logging.error(
            f"Pyannote diarization error for '{audio_path}': {e}", exc_info=True
        )
        return []  # Return empty list on error to allow pipeline to continue gracefully


def format_diarization_for_llm(diarization_result: List[Dict[str, Any]]) -> str:
    """
    Formats diarization results into a string suitable for an LLM prompt.

    Args:
        diarization_result: The list of speaker segments from `diarize_speakers`.

    Returns:
        A formatted string representing speaker turns, or a message indicating
        that diarization is unavailable.
    """
    if not diarization_result:
        return "Speaker diarization information is not available.\n"

    output = "Speaker Turns (Format: SPEAKER_ID StartTime[H:MM:SS.ms]-EndTime[H:MM:SS.ms]):\n"
    for segment in diarization_result:
        try:
            # Format timestamps precisely using timedelta
            start_td = datetime.timedelta(seconds=segment["start"])
            end_td = datetime.timedelta(seconds=segment["end"])
            # Pad microseconds to 3 digits (milliseconds) and handle potential rounding errors
            start_ms = str(start_td.microseconds // 1000).zfill(3)
            end_ms = str(end_td.microseconds // 1000).zfill(3)
            start_str = f"{str(start_td).split('.')[0]}.{start_ms}"
            end_str = f"{str(end_td).split('.')[0]}.{end_ms}"

            output += f"- {segment['speaker']} {start_str}-{end_str}\n"
        except Exception as e:
            logging.warning(
                f"Error formatting segment {segment}: {e}. Skipping segment."
            )
            continue  # Skip malformed segments
    return output


def correct_transcript(
    raw_transcript: str, diarization_result: List[Dict[str, Any]], correction_model: str
) -> str:
    """
    Corrects the transcript using a specified LLM via OpenRouter, incorporating diarization info.

    Args:
        raw_transcript: The raw text output from the transcription model.
        diarization_result: The list of speaker segments from `diarize_speakers`.
        correction_model: The identifier of the LLM to use for correction on OpenRouter.

    Returns:
        The corrected and formatted transcript text.

    Raises:
        ConnectionError: If the OpenAI client is not initialized (missing API key).
        RuntimeError: If the correction API call fails.
    """
    if not client:
        raise ConnectionError(
            "OpenAI client not initialized. Check OPENROUTER_API_KEY."
        )
    if not raw_transcript:
        logging.warning("Raw transcript is empty, cannot perform correction.")
        return ""  # Return empty string if nothing to correct

    logging.info(
        f"Starting transcript correction using model '{correction_model}' via OpenRouter..."
    )

    # Format diarization data for the prompt
    speaker_turns_text = format_diarization_for_llm(diarization_result)

    # --- Develop the Prompt Template ---
    # This prompt is crucial and likely needs refinement based on testing results.
    # Consider making this configurable or loading from a file.
    prompt_template = f"""You are an expert transcript editor. Your task is to take a raw, unprocessed transcript generated by an ASR system and speaker turn information generated by a diarization system, and produce a clean, accurate, and highly readable final transcript.

**Input:**
1.  **Speaker Turns:** A list indicating which speaker (e.g., SPEAKER_00, SPEAKER_01) was speaking during specific time intervals. This information might be approximate.
2.  **Raw Transcript:** The potentially inaccurate output from the ASR system.

**Instructions:**
1.  **Integrate Speaker Labels:** Use the "Speaker Turns" data to assign the correct speaker label (e.g., "SPEAKER_00:", "SPEAKER_01:") to the corresponding speech segments in the transcript. Start each new speaker's turn on a new line, preceded by their label. Use your best judgment to align the text with the speaker turns, even if the exact word boundaries are slightly off.
2.  **Correct ASR Errors:** Fix spelling mistakes, grammatical errors, and punctuation inaccuracies commonly found in raw ASR output. Ensure proper capitalization (start of sentences, proper nouns).
3.  **Improve Readability:** Break down long paragraphs into shorter ones where appropriate. Ensure smooth transitions between speaker turns. Remove filler words (um, uh, like) *only if* they significantly detract from readability, but generally preserve the natural flow of speech.
4.  **Handle Missing/Incorrect Diarization:** If the "Speaker Turns" section indicates information is unavailable or seems clearly wrong based on the transcript content, format the text as a single coherent block without speaker labels. Add a note at the beginning like "[Note: Speaker diarization data was unavailable or inconsistent.]" if this occurs.
5.  **Accuracy is Key:** Preserve the original meaning and intent of the speech. Do not add information not present in the raw transcript.
6.  **Output Format:** Produce *only* the final, corrected, speaker-labeled transcript text. Do NOT include the speaker turn list, timestamps, or any other meta-commentary in the final output unless specifically instructed (e.g., the note about missing diarization).

**Speaker Turns:**
{speaker_turns_text}

**Raw Transcript:**
{raw_transcript}

**Corrected and Formatted Transcript:**
"""

    try:
        logging.info(
            f"Sending correction request to OpenRouter model: {correction_model}"
        )
        # Use a model known for strong instruction following and text generation.
        # Increase timeout for potentially long correction tasks
        chat_completion = client.with_options(timeout=300.0).chat.completions.create(
            model=correction_model,  # e.g., Claude 3 Haiku/Sonnet, GPT-4 Turbo
            messages=[
                # System message can help set the persona more strongly
                {
                    "role": "system",
                    "content": "You are an expert transcript editor focused on accuracy and readability, integrating speaker labels based on provided turn information.",
                },
                {"role": "user", "content": prompt_template},
            ],
            temperature=0.5,  # Lower temperature for more deterministic corrections
            # max_tokens= ... # Consider setting if needed, but often inferred
        )
        corrected_transcript = chat_completion.choices[0].message.content.strip()
        if not corrected_transcript:
            # Handle cases where the model returns an empty string
            logging.warning(
                f"Correction model '{correction_model}' returned an empty result."
            )
            # Depending on requirements, either return empty or raise error
            # Returning empty for now to avoid breaking the pipeline completely
            return ""
            # raise RuntimeError(f"Correction model '{correction_model}' returned an empty result.")

        logging.info("Correction successful.")
        return corrected_transcript
    except Exception as e:
        logging.error(
            f"OpenRouter correction API error using model {correction_model}: {e}",
            exc_info=True,
        )
        raise RuntimeError(f"Correction failed: {e}") from e


# --- Main Processing Function ---


def process_audio_file(
    audio_path: str,
    output_dir: str,
    transcription_model: str,
    correction_model: str,
    summarization_model: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Processes a single audio file through the full pipeline:
    Transcription -> Diarization -> Correction -> Summarization (required).

    Saves the final corrected transcript and summary to the output directory.

    Args:
        audio_path: Path to the input audio file (WAV recommended).
        output_dir: Directory where the corrected transcript should be saved.
        transcription_model: Model identifier for transcription.
        correction_model: Model identifier for correction.
        summarization_model: Model identifier for summarization. Required.

    Returns:
        A tuple containing:
        - The absolute path to the saved corrected transcript file (or None if failed).
        - The absolute path to the saved summary file (or None if failed).

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        ConnectionError: If the OpenAI client is not initialized.
        RuntimeError: If any step in the processing pipeline fails.
        ValueError: If required parameters are missing or invalid.
    """
    # Validate inputs
    if not os.path.exists(audio_path):
        error_msg = f"Input audio file not found: {audio_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not output_dir or not isinstance(output_dir, str):
        error_msg = "Invalid output directory specified."
        logging.error(error_msg)
        raise ValueError(error_msg)

    if not summarization_model:
        error_msg = "Summarization model is required but was not provided."
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Create a report dictionary to track processing details
    report = {
        "audio_path": audio_path,
        "output_dir": output_dir,
        "start_time": datetime.datetime.now().isoformat(),
        "steps": {},
        "success": False,
        "error": None,
        "system_info": get_system_info(),
        "backups": {},
        "recovery_file": None,
    }

    logging.info(f"Starting full processing for audio file: {audio_path}")
    corrected_transcript_path: Optional[str] = None
    summary_path: Optional[str] = None
    corrected_transcript_content: Optional[str] = None

    try:
        # Log environment and system info at the start of processing
        logging.debug(f"Processing audio file: {audio_path}")
        logging.debug(f"Output directory: {output_dir}")
        logging.debug(f"Python executable: {sys.executable}")
        logging.debug(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
        logging.debug(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")

        # Log system information
        logging.debug("System Information:")
        logging.debug(f"OS: {platform.system()} {platform.release()}")
        logging.debug(f"Python version: {sys.version}")
        logging.debug(f"CPU count: {os.cpu_count()}")
        logging.debug(
            f"Available memory: {psutil.virtual_memory().available / (1024 * 1024):.2f} MB"
        )

        # Log installed packages
        try:
            import pkg_resources

            installed_packages = [
                f"{dist.key} {dist.version}" for dist in pkg_resources.working_set
            ]
            logging.debug("Installed packages:")
            for pkg in installed_packages:
                logging.debug(f"  {pkg}")
        except Exception as e:
            logging.warning(f"Could not list installed packages: {e}")

        # --- Step 1: Transcribe Audio ---
        step_start = datetime.datetime.now()
        logging.info("--- Step 1: Transcribing audio ---")
        report["steps"]["transcription"] = {"start_time": step_start.isoformat()}

        try:
            raw_transcript = transcribe_audio(audio_path, transcription_model)
            if not raw_transcript:
                error_context = {
                    "step": "transcription",
                    "audio_path": audio_path,
                    "model": transcription_model,
                }
                error = TranscriptionError(
                    "Transcription returned empty result", error_context
                )
                error_diagnosis = diagnose_error(error, error_context)

                report["steps"]["transcription"]["success"] = False
                report["steps"]["transcription"]["error"] = str(error)
                report["steps"]["transcription"]["diagnosis"] = error_diagnosis
                report["steps"]["transcription"][
                    "end_time"
                ] = datetime.datetime.now().isoformat()
                report["error"] = str(error)
                report["error_diagnosis"] = error_diagnosis

                # Create backups before cleanup
                report["backups"] = backup_intermediate_files(audio_path, output_dir)
                _cleanup_intermediate_audio(audio_path)

                # Create recovery file
                report["recovery_file"] = create_error_recovery_file(
                    error_diagnosis, output_dir
                )
                _save_report(report, output_dir)
                return None, None

            report["steps"]["transcription"]["success"] = True
            report["steps"]["transcription"]["length"] = len(raw_transcript)
            report["steps"]["transcription"][
                "end_time"
            ] = datetime.datetime.now().isoformat()
            logging.info(
                f"Transcription successful. Length: {len(raw_transcript)} characters"
            )

        except Exception as e:
            error_context = {
                "step": "transcription",
                "audio_path": audio_path,
                "model": transcription_model,
            }
            error_diagnosis = diagnose_error(e, error_context)

            report["steps"]["transcription"]["success"] = False
            report["steps"]["transcription"]["error"] = str(e)
            report["steps"]["transcription"]["diagnosis"] = error_diagnosis
            report["steps"]["transcription"][
                "end_time"
            ] = datetime.datetime.now().isoformat()
            report["error"] = str(e)
            report["error_diagnosis"] = error_diagnosis

            # Create backups before cleanup
            report["backups"] = backup_intermediate_files(audio_path, output_dir)
            _cleanup_intermediate_audio(audio_path)

            # Create recovery file
            report["recovery_file"] = create_error_recovery_file(
                error_diagnosis, output_dir
            )
            _save_report(report, output_dir)

            if isinstance(e, (FileNotFoundError, PermissionError)):
                raise
            else:
                raise TranscriptionError(
                    f"Transcription failed: {str(e)}", {"original_error": str(e)}
                )

        # --- Step 2: Diarize Speakers (Required step) ---
        step_start = datetime.datetime.now()
        logging.info("--- Step 2: Diarizing speakers ---")
        report["steps"]["diarization"] = {"start_time": step_start.isoformat()}

        try:
            diarization_result = diarize_speakers(audio_path)
            if not diarization_result:
                error_context = {"step": "diarization", "audio_path": audio_path}
                error = DiarizationError(
                    "Diarization returned empty result. This step is required.",
                    error_context,
                )
                error_diagnosis = diagnose_error(error, error_context)

                report["steps"]["diarization"]["success"] = False
                report["steps"]["diarization"]["error"] = str(error)
                report["steps"]["diarization"]["diagnosis"] = error_diagnosis
                report["steps"]["diarization"][
                    "end_time"
                ] = datetime.datetime.now().isoformat()
                report["error"] = str(error)
                report["error_diagnosis"] = error_diagnosis

                # Create backups before cleanup
                report["backups"] = backup_intermediate_files(audio_path, output_dir)
                _cleanup_intermediate_audio(audio_path)

                # Create recovery file
                report["recovery_file"] = create_error_recovery_file(
                    error_diagnosis, output_dir
                )
                _save_report(report, output_dir)
                return None, None

            report["steps"]["diarization"]["success"] = True
            report["steps"]["diarization"]["segments_count"] = len(diarization_result)
            report["steps"]["diarization"][
                "end_time"
            ] = datetime.datetime.now().isoformat()
            logging.info(
                f"Diarization successful. Found {len(diarization_result)} speaker segments."
            )

        except Exception as e:
            error_context = {"step": "diarization", "audio_path": audio_path}
            error_diagnosis = diagnose_error(e, error_context)

            report["steps"]["diarization"]["success"] = False
            report["steps"]["diarization"]["error"] = str(e)
            report["steps"]["diarization"]["diagnosis"] = error_diagnosis
            report["steps"]["diarization"][
                "end_time"
            ] = datetime.datetime.now().isoformat()
            report["error"] = str(e)
            report["error_diagnosis"] = error_diagnosis

            # Create backups before cleanup
            report["backups"] = backup_intermediate_files(audio_path, output_dir)
            _cleanup_intermediate_audio(audio_path)

            # Create recovery file
            report["recovery_file"] = create_error_recovery_file(
                error_diagnosis, output_dir
            )
            _save_report(report, output_dir)
            raise DiarizationError(
                f"Diarization failed: {str(e)}", {"original_error": str(e)}
            )

        # Continue with the rest of the steps using the same pattern...

        # ... rest of the code remains the same ...

        # --- Step 6: Cleanup intermediate files ---
        step_start = datetime.datetime.now()
        logging.info("--- Step 6: Cleaning up intermediate files ---")
        report["steps"]["cleanup"] = {"start_time": step_start.isoformat()}

        try:
            # First clean up intermediate audio files
            _cleanup_intermediate_audio(audio_path)

            # Then enforce the strict output structure
            video_filename = os.path.basename(audio_path).replace(".wav", ".mp4")
            if not video_filename.endswith(".mp4"):
                video_filename += ".mp4"

            enforce_result = enforce_output_structure(output_dir, video_filename)

            if enforce_result:
                report["steps"]["cleanup"]["success"] = True
                report["steps"]["cleanup"]["structure_enforced"] = True
                report["steps"]["cleanup"][
                    "end_time"
                ] = datetime.datetime.now().isoformat()
                logging.info(
                    "Cleanup of intermediate files and structure enforcement completed successfully."
                )
            else:
                report["steps"]["cleanup"]["success"] = False
                report["steps"]["cleanup"]["structure_enforced"] = False
                report["steps"]["cleanup"][
                    "error"
                ] = "Failed to enforce output structure"
                report["steps"]["cleanup"][
                    "end_time"
                ] = datetime.datetime.now().isoformat()
                logging.warning(
                    "Cleanup of intermediate files succeeded but structure enforcement failed."
                )
        except Exception as e:
            error_msg = (
                f"Cleanup of intermediate audio files encountered an error: {str(e)}"
            )
            logging.warning(error_msg, exc_info=True)
            report["steps"]["cleanup"]["success"] = False
            report["steps"]["cleanup"]["error"] = error_msg
            report["steps"]["cleanup"]["end_time"] = datetime.datetime.now().isoformat()
            # Continue despite cleanup errors

        # --- Final Success ---
        report["success"] = True
        report["end_time"] = datetime.datetime.now().isoformat()
        _save_report(report, output_dir)

        return corrected_transcript_path, summary_path

    except Exception as unexpected_err:
        error_msg = f"Unexpected error processing audio file {audio_path}: {str(unexpected_err)}"
        logging.error(error_msg, exc_info=True)
        report["success"] = False
        report["error"] = error_msg
        report["end_time"] = datetime.datetime.now().isoformat()
        _save_report(report, output_dir)
        raise RuntimeError(error_msg) from unexpected_err


def enforce_output_structure(output_dir: str, video_filename: str) -> bool:
    """
    Enforces the strict output structure rule: each video folder should only contain
    1. The original video file
    2. The transcript from the video
    3. The transcript summary

    Args:
        output_dir: Path to the video directory
        video_filename: Filename of the original video (without path)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
            logging.error(f"Video directory does not exist: {output_dir}")
            return False

        base_name = os.path.splitext(video_filename)[0]

        # Define allowed files
        allowed_files = [
            video_filename,  # Original video
            f"{base_name}_corrected.md",  # Transcript
            f"{base_name}_summary.md",  # Summary
            f"{base_name}_report.json",  # Processing report (allowed but optional)
        ]

        removed_items = []

        # First, remove all subdirectories
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)

            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    removed_items.append(("directory", item))
                    logging.info(f"Removed directory: {item_path}")
                except Exception as e:
                    logging.error(f"Failed to remove directory: {item_path}: {e}")
                    return False

        # Then remove all files except allowed ones
        for item in os.listdir(output_dir):
            if item not in allowed_files:
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    try:
                        os.remove(item_path)
                        removed_items.append(("file", item))
                        logging.info(f"Removed file: {item_path}")
                    except Exception as e:
                        logging.error(f"Failed to remove file: {item_path}: {e}")
                        return False

        # Log cleanup summary
        if removed_items:
            dirs_removed = sum(
                1 for item_type, _ in removed_items if item_type == "directory"
            )
            files_removed = sum(
                1 for item_type, _ in removed_items if item_type == "file"
            )
            logging.info(
                f"Structure cleanup summary: Removed {dirs_removed} directories and {files_removed} files"
            )
            for item_type, name in removed_items:
                logging.debug(f"  - Removed {item_type}: {name}")
        else:
            logging.info("No items needed to be removed, structure was already correct")

        # Verify the final structure
        remaining_files = os.listdir(output_dir)
        if all(f in allowed_files for f in remaining_files):
            logging.info(
                f"Output structure successfully enforced. Remaining files: {remaining_files}"
            )
            return True
        else:
            unexpected_files = [f for f in remaining_files if f not in allowed_files]
            logging.error(
                f"Failed to enforce output structure. Unexpected files remain: {unexpected_files}"
            )
            return False

    except Exception as e:
        logging.error(f"Error enforcing output structure: {e}", exc_info=True)
        return False


def _cleanup_intermediate_audio(audio_path: str):
    """
    Removes intermediate audio files (e.g., *_audio_attempt*.wav) and other temporary files
    associated with the given audio_path. Ensures that only the final output structure remains clean.
    """
    try:
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Log cleanup start
        logging.info(f"Starting cleanup of intermediate audio files for {base_name}")

        # Remove files matching *_audio_attempt*.wav in the same directory
        removed_files = []
        for fname in os.listdir(base_dir):
            if (
                fname.startswith(base_name)
                and "audio_attempt" in fname
                and fname.endswith(".wav")
            ):
                file_path = os.path.join(base_dir, fname)
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    os.remove(file_path)
                    removed_files.append((fname, file_size))
                    logging.info(
                        f"Removed intermediate audio file: {file_path} ({file_size:.2f} MB)"
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to remove intermediate audio file: {file_path}: {e}"
                    )

        # Log cleanup summary
        if removed_files:
            total_size = sum(size for _, size in removed_files)
            logging.info(
                f"Cleanup summary: Removed {len(removed_files)} files, total size: {total_size:.2f} MB"
            )
            for fname, size in removed_files:
                logging.debug(f"  - {fname}: {size:.2f} MB")
        else:
            logging.info("No intermediate audio files found to remove")

    except Exception as cleanup_err:
        logging.warning(
            f"Cleanup of intermediate audio files failed: {cleanup_err}", exc_info=True
        )


def _save_report(report: Dict[str, Any], output_dir: str) -> None:
    """
    Saves the processing report to a JSON file in both the output directory and logs directory.

    Args:
        report: Dictionary containing processing details and results
        output_dir: Directory where the report should be saved
    """
    try:
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Create report filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(report.get("audio_path", "unknown"))
        report_filename = f"{os.path.splitext(base_name)[0]}_report.json"
        logs_report_filename = (
            f"logs/report_{os.path.splitext(base_name)[0]}_{timestamp}.json"
        )

        # Save report to output directory
        output_report_path = os.path.join(output_dir, report_filename)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.info(f"Processing report saved to: {output_report_path}")

        # Save report to logs directory
        with open(logs_report_filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logging.info(f"Processing report saved to logs: {logs_report_filename}")

        # Log report summary
        logging.info("Report summary:")
        logging.info(f"  Success: {report.get('success', False)}")
        if not report.get("success", False) and report.get("error"):
            logging.info(f"  Error: {report.get('error')}")

        logging.info("  Steps:")
        for step_name, step_data in report.get("steps", {}).items():
            success = step_data.get("success", False)
            status = "Success" if success else "Failed"
            logging.info(f"    - {step_name}: {status}")
            if not success and step_data.get("error"):
                logging.info(f"      Error: {step_data.get('error')}")

    except Exception as e:
        logging.error(f"Failed to save processing report: {str(e)}", exc_info=True)

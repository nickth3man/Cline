import os
import subprocess
import logging
from openai import OpenAI
from pyannote.audio import Pipeline
import torch
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize OpenAI Client
if OPENROUTER_API_KEY:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    logging.info("OpenAI client configured for OpenRouter.")
else:
    client = None
    logging.warning(
        "OPENROUTER_API_KEY not found in environment variables. API calls will fail."
    )

# Load Diarization Pipeline
diarization_pipeline = None
try:
    if HF_TOKEN is None:
        logging.warning(
            "HF_TOKEN environment variable not set. Trying to load pyannote pipeline without auth token. This may fail."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device} for pyannote.audio")

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN
    )
    diarization_pipeline.to(torch.device(device))
    logging.info("Pyannote diarization pipeline loaded successfully.")
except Exception as e:
    logging.error(
        f"Failed to load pyannote pipeline: {e}. Diarization will be skipped.",
        exc_info=True,
    )
    diarization_pipeline = None


def check_ffmpeg():
    """Checks if ffmpeg is accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        logging.info("FFmpeg found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning(
            "FFmpeg not found or not executable. Audio extraction/conversion might fail."
        )
        return False


_ffmpeg_present = check_ffmpeg()


def extract_or_convert_audio(input_path: str, output_wav_path: str) -> str:
    """Extracts audio from video or converts to WAV (16kHz mono) using FFmpeg."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not _ffmpeg_present:
        raise RuntimeError(
            "FFmpeg is required for audio preparation but was not found."
        )

    logging.info(
        f"Converting/Extracting audio to 16kHz mono WAV: {input_path} -> {output_wav_path}"
    )
    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # Standard WAV codec
        "-ac",
        "1",  # Mono channel
        "-ar",
        "16000",  # 16kHz sample rate
        "-nostdin",  # Prevent interference with stdin (good practice)
        "-y",  # Overwrite output file if it exists
        output_wav_path,
    ]
    try:
        process = subprocess.run(command, check=True, capture_output=True)
        logging.info(f"FFmpeg conversion successful: {output_wav_path}")
        return output_wav_path
    except subprocess.CalledProcessError as e:
        error_message = (
            f"FFmpeg failed for {input_path}: {e.stderr.decode(errors='ignore')}"
        )
        logging.error(error_message)
        raise RuntimeError(error_message)
    except FileNotFoundError:
        logging.error("FFmpeg command not found during execution.")
        raise RuntimeError(
            "FFmpeg command not found. Please ensure FFmpeg is installed and in your PATH."
        )
    except Exception as e:
        error_message = f"An unexpected error occurred during FFmpeg execution for {input_path}: {str(e)}"
        logging.error(error_message, exc_info=True)
        raise RuntimeError(error_message)


def transcribe_audio(audio_path: str) -> str:
    """Transcribes the given audio file using Whisper via OpenRouter API.
    If the file is too large, splits it into 15-minute chunks and concatenates the results.
    """
    import wave
    import math
    import tempfile
    import shutil

    if not client:
        raise ConnectionError(
            "OpenAI client not initialized. Check OPENROUTER_API_KEY."
        )
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Helper to get duration in seconds
    def get_wav_duration(path):
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)

    max_chunk_seconds = (
        2 * 60
    )  # 2 minutes, to ensure chunks are well under 25MB Whisper API limit
    duration = get_wav_duration(audio_path)
    logging.info(f"Audio duration: {duration:.2f} seconds")

    if duration <= max_chunk_seconds:
        # Short enough, transcribe directly
        logging.info(f"Starting transcription for {audio_path} via OpenRouter...")
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = client.with_options(
                    timeout=600.0
                ).audio.transcriptions.create(
                    model="openai/whisper-large-v3", file=audio_file
                )
            logging.info(f"Transcription successful for {audio_path}.")
            return transcript.text
        except Exception as e:
            logging.error(
                f"OpenRouter transcription API error for {audio_path}: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Transcription failed: {e}") from e

    # Too long, split into chunks
    logging.info(
        f"Audio is longer than {max_chunk_seconds} seconds. Splitting into chunks."
    )
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    chunk_paths = []
    try:
        # Use ffmpeg to split into 15-minute chunks
        chunk_pattern = os.path.join(temp_dir, "chunk_%03d.wav")
        command = [
            "ffmpeg",
            "-i",
            audio_path,
            "-f",
            "segment",
            "-segment_time",
            str(max_chunk_seconds),
            "-c",
            "copy",
            chunk_pattern,
            "-y",
        ]
        subprocess.run(command, check=True, capture_output=True)
        chunk_paths = sorted(
            [
                os.path.join(temp_dir, f)
                for f in os.listdir(temp_dir)
                if f.startswith("chunk_") and f.endswith(".wav")
            ]
        )
        if not chunk_paths:
            raise RuntimeError("FFmpeg did not produce any audio chunks.")

        full_transcript = ""
        for idx, chunk in enumerate(chunk_paths):
            logging.info(f"Transcribing chunk {idx+1}/{len(chunk_paths)}: {chunk}")
            try:
                with open(chunk, "rb") as audio_file:
                    transcript = client.with_options(
                        timeout=600.0
                    ).audio.transcriptions.create(
                        model="openai/whisper-large-v3", file=audio_file
                    )
                full_transcript += transcript.text + "\n"
            except Exception as e:
                logging.error(
                    f"Transcription failed for chunk {chunk}: {e}", exc_info=True
                )
                full_transcript += f"\n[Transcription failed for chunk {idx+1}: {e}]\n"
        logging.info(f"All chunks transcribed and concatenated for {audio_path}.")
        return full_transcript.strip()
    except Exception as e:
        logging.error(f"Error during chunked transcription: {e}", exc_info=True)
        raise RuntimeError(f"Chunked transcription failed: {e}") from e
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def diarize_speakers(audio_path: str) -> list:
    """Performs speaker diarization using the local pyannote.audio pipeline."""
    if not diarization_pipeline:
        logging.warning(
            f"Diarization pipeline not loaded. Skipping diarization for {audio_path}."
        )
        return []
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found for diarization: {audio_path}")

    logging.info(f"Starting speaker diarization for {audio_path}...")
    try:
        diarization = diarization_pipeline(audio_path, num_speakers=None)

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
            )

        segments.sort(key=lambda x: x["start"])
        logging.info(
            f"Diarization successful for {audio_path}. Found {len(segments)} segments for {len(diarization.labels())} speakers."
        )
        return segments
    except Exception as e:
        logging.error(
            f"Pyannote diarization error for {audio_path}: {e}", exc_info=True
        )
        return []


def format_diarization_for_llm(diarization_result: list) -> str:
    """Formats diarization results into a string suitable for an LLM prompt."""
    if not diarization_result:
        return "Speaker diarization information is not available.\n"

    output = "Speaker Turns (Format: SPEAKER_ID StartTime[H:MM:SS.ms]-EndTime[H:MM:SS.ms]):\n"
    for segment in diarization_result:
        start_td = datetime.timedelta(seconds=segment["start"])
        end_td = datetime.timedelta(seconds=segment["end"])
        start_str = f"{str(start_td).split('.')[0]}.{str(start_td.microseconds // 1000).zfill(3)}"
        end_str = (
            f"{str(end_td).split('.')[0]}.{str(end_td.microseconds // 1000).zfill(3)}"
        )

        output += f"- {segment['speaker']} {start_str}-{end_str}\n"
    return output


def correct_transcript(
    raw_transcript: str,
    diarization_result: list,
    correction_model: str = "anthropic/claude-3-haiku-20240307",
) -> str:
    """Corrects the transcript using an LLM via OpenRouter, incorporating diarization info."""
    if not client:
        raise ConnectionError(
            "OpenAI client not initialized. Check OPENROUTER_API_KEY."
        )
    if not raw_transcript:
        logging.warning("Raw transcript is empty, cannot perform correction.")
        return ""

    logging.info(
        f"Starting transcript correction using model {correction_model} via OpenRouter..."
    )

    speaker_turns_text = format_diarization_for_llm(diarization_result)

    prompt_template = f"""You are an expert transcript editor. Your task is to take a raw, unprocessed transcript generated by an ASR system and speaker turn information generated by a diarization system, and produce a clean, accurate, and highly readable final transcript.

**Input:**
1.  **Speaker Turns:** A list indicating which speaker (e.g., SPEAKER_00, SPEAKER_01) was speaking during specific time intervals.
2.  **Raw Transcript:** The potentially inaccurate output from the ASR system.

**Instructions:**
1.  **Integrate Speaker Labels:** Use the "Speaker Turns" data to assign the correct speaker label (e.g., "SPEAKER_00:", "SPEAKER_01:") to the corresponding speech segments in the transcript. Start each new speaker's turn on a new line, preceded by their label.
2.  **Correct ASR Errors:** Fix spelling mistakes, grammatical errors, and punctuation inaccuracies commonly found in raw ASR output. Pay close attention to capitalization (start of sentences, proper nouns).
3.  **Improve Readability:** Break down long paragraphs into shorter ones where appropriate. Ensure smooth transitions between speaker turns. Remove filler words (um, uh, like) *only if* they significantly detract from readability, but generally preserve the natural flow of speech.
4.  **Handle Overlap/Uncertainty (Implicitly):** The raw transcript and speaker turns might not align perfectly. Use your best judgment to segment the text according to the speaker turn information, even if the exact word boundaries are slightly off. If the diarization is empty or clearly wrong, format the text as a single block without speaker labels.
5.  **Accuracy is Key:** Preserve the original meaning. Do not add information not present in the raw transcript.
6.  **Output Format:** Produce *only* the final, corrected, speaker-labeled transcript text. Do NOT include the speaker turn list, timestamps, or any other meta-commentary in the final output.

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
        chat_completion = client.with_options(timeout=300.0).chat.completions.create(
            model=correction_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert transcript editor focused on accuracy and readability.",
                },
                {"role": "user", "content": prompt_template},
            ],
            temperature=0.5,
        )
        corrected_transcript = chat_completion.choices[0].message.content.strip()
        logging.info("Transcript correction successful.")
        return corrected_transcript
    except Exception as e:
        logging.error(f"OpenRouter correction API error: {e}", exc_info=True)
        raise RuntimeError(f"Transcript correction failed: {e}") from e

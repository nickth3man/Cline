import pytest
import os
import torch
from src.utils.workflow_logic import transcribe_audio, assign_speakers_to_segments # Import assign_speakers_to_segments
import pandas as pd # Import pandas for DataFrame check
import whisperx # Import whisperx for load_audio

# Test function for WhisperX integration
def test_whisperx_transcription_and_diarization():
    """Tests the integration of WhisperX for transcription and diarization using a sample audio file."""
    # Ensure HF_TOKEN is set for diarization
    if not os.getenv("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set, skipping diarization test.")

    audio_path = "sample/harvard.wav"
    if not os.path.exists(audio_path):
        pytest.skip(f"Sample audio file not found at {audio_path}, skipping test.")

    # Perform transcription using WhisperX
    try:
        # Added a dummy video_id as required by the function signature
        transcription_result = transcribe_audio("test_video_id", audio_path)
    except Exception as e:
        pytest.fail(f"Transcription failed with WhisperX: {e}")

    # Assert that transcription returned a dictionary with segments
    assert isinstance(transcription_result, dict)
    assert "segments" in transcription_result
    assert isinstance(transcription_result["segments"], list)
    assert len(transcription_result["segments"]) > 0

    # Extract concatenated text for a basic check
    transcribed_text = " ".join([segment["text"] for segment in transcription_result["segments"]])
    assert len(transcribed_text) > 0

    # Perform diarization and speaker assignment using WhisperX
    # The transcribe_audio function now handles diarization internally,
    # so we just need to check the structure of the returned segments.
    diarization_segments = transcription_result.get("segments", [])

    # Assert that diarization returned a list of segments
    assert isinstance(diarization_segments, list)
    # Assert that segments were produced
    assert len(diarization_segments) > 0, "No diarization segments were produced."

    # If segments are returned, assert that they have expected keys and speaker labels
    speakers_assigned = False
    for segment in diarization_segments:
        assert "speaker" in segment
        assert isinstance(segment["speaker"], str)
        assert "text" in segment
        assert isinstance(segment["text"], str)
        assert "start" in segment
        assert isinstance(segment["start"], (int, float))
        assert "end" in segment
        assert isinstance(segment["end"], (int, float))
        if segment["speaker"] != "Unknown Speaker":
            speakers_assigned = True

    # Assert that at least one speaker was assigned (assuming harvard.wav has speakers)
    assert speakers_assigned, "No speakers were assigned to any segments."

    print(f"Transcription and Diarization successful for {audio_path}. {len(diarization_segments)} segments with speaker labels.")

# Note: Additional tests for different audio lengths and edge cases can be added here.
# For cumulative testing, running pytest on the entire tests/ directory is sufficient.
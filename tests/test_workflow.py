# --- START OF FILE tests/test_workflow.py ---

import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock, mock_open, call, ANY # Import call and ANY
from pydub import AudioSegment # Import AudioSegment for creating dummy audio files
import io # Import io for BytesIO
import json # Import json to load diarization results

# Mock environment variables before importing the module under test
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-fakekey")
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    # Import the module *after* setting env vars
    global workflow_logic
    from src.utils import workflow_logic

    # Reload to ensure mocks are applied if already imported
    import importlib

    importlib.reload(workflow_logic)


# --- Fixtures ---


@pytest.fixture
def sample_diarization_result():
    # This fixture is for the old diarization method and is no longer directly used
    # but keeping it for reference if needed for mocking the combined output structure
    return [
        {"speaker": "SPEAKER_00", "start": 0.500, "end": 2.800},
        {"speaker": "SPEAKER_01", "start": 3.100, "end": 5.200},
    ]


@pytest.fixture
def mock_openai_client():
    """Provides a mock OpenAI client for testing API interactions (primarily correction/summarization)."""
    with patch("src.utils.workflow_logic.client") as mock_client:
        # Mock the chat completion (correction and summarization) response
        mock_chat_choice = MagicMock()
        # Mock content for correction - should include speaker labels
        mock_chat_choice.message.content = """SPEAKER_01: This is the corrected first segment.
SPEAKER_02: This is the corrected second segment."""
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [mock_chat_choice]
        mock_client.chat.completions.create.return_value = (
            mock_chat_completion
        )

        yield mock_client


@pytest.fixture
def mock_ffmpeg():
    """Fixture to mock ffmpeg subprocess calls"""
    with patch("src.utils.workflow_logic.subprocess.run") as mock_run:
        # Mock the successful return for version check
        mock_process = MagicMock()
        mock_process.stderr = b""
        mock_process.stdout = b"ffmpeg version ..."
        mock_run.return_value = mock_process
        yield mock_run

# Mock WhisperX and Pyannote components used in transcribe_audio
@pytest.fixture
def mock_whisperx_components():
    with patch("src.utils.workflow_logic.whisperx") as mock_whisperx, \
         patch("src.utils.workflow_logic.initialize_diarization_pipeline") as mock_init_diarization:

        # Mock whisperx.load_audio
        mock_whisperx.load_audio.return_value = MagicMock() # Return a dummy audio object

        # Mock whisperx.load_model
        mock_whisperx.load_model.return_value = MagicMock() # Return a dummy model

        # Mock whisperx.load_align_model
        mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock()) # Return dummy model and metadata

        # Mock whisperx.DiarizationPipeline
        mock_whisperx.DiarizationPipeline.return_value = MagicMock() # Return a dummy diarization pipeline

        # Mock whisperx.align
        # This should return a structure similar to the transcription result but with aligned segments
        mock_whisperx.align.return_value = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "words": [{"word": "hello", "start": 0.0, "end": 1.0}]},
                {"start": 1.5, "end": 2.5, "text": "world", "words": [{"word": "world", "start": 1.5, "end": 2.5}]},
            ],
            "language": "en"
        }

        # Mock the pyannote.audio Pipeline call within initialize_diarization_pipeline
        mock_pyannote_pipeline_instance = MagicMock()
        # Mock the __call__ method of the pyannote pipeline instance
        # This should return a pyannote Annotation object or similar structure
        # Create a simple mock object that explicitly has an itertracks method
        mock_pyannote_annotation = MagicMock()
        # Simulate the itertracks method returning speaker turns with clear overlap
        mock_pyannote_annotation.itertracks.return_value = [
            (MagicMock(start=0.0, end=1.5), "segment_0", "SPEAKER_01"), # Overlaps with segment 1
            (MagicMock(start=1.0, end=3.0), "segment_1", "SPEAKER_02"), # Overlaps with segment 2
        ]
        mock_pyannote_pipeline_instance.return_value = mock_pyannote_annotation
        mock_init_diarization.return_value = mock_pyannote_pipeline_instance


        # Mock whisperx_model.transcribe
        mock_whisperx.whisperx_model = MagicMock() # Access the module-level variable
        mock_whisperx.whisperx_model.transcribe.return_value = {
             "segments": [ # Raw segments from transcription before alignment
                {"start": 0.0, "end": 1.1, "text": "hello"},
                {"start": 1.4, "end": 2.6, "text": "world"},
            ],
            "language": "en"
        }


        yield mock_whisperx, mock_init_diarization


# --- Test Cases ---


def test_check_ffmpeg_found(mock_ffmpeg):
    """Test check_ffmpeg when ffmpeg is found"""
    workflow_logic.check_ffmpeg()
    mock_ffmpeg.assert_called_once_with(
        ["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def test_check_ffmpeg_not_found(mock_ffmpeg):
    """Test check_ffmpeg when ffmpeg is not found"""
    mock_ffmpeg.side_effect = FileNotFoundError
    with pytest.raises(EnvironmentError, match="FFmpeg is required but not found."):
        workflow_logic.check_ffmpeg()
    mock_ffmpeg.assert_called_once_with(
        ["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def test_transcribe_audio_success(mock_whisperx_components, tmp_path):
    """Test successful transcription and diarization with transcribe_audio"""
    audio_file = tmp_path / "audio.wav"
    # Create a small, valid dummy WAV file using pydub
    # Removed channels=1 and sample_width=2 as AudioSegment.silent does not accept them
    silent_audio = AudioSegment.silent(duration=5000, frame_rate=16000)
    silent_audio.export(audio_file, format="wav")

    video_id = "test_video_id"

    # Call the transcribe_audio function
    transcription_result = workflow_logic.transcribe_audio(video_id, str(audio_file))
    
    # Assertions for the combined result structure
    assert isinstance(transcription_result, dict)
    assert "segments" in transcription_result
    assert isinstance(transcription_result["segments"], list)
    assert len(transcription_result["segments"]) > 0 # Should have segments if mocks are working

    # Check for speaker labels in segments
    speakers_assigned = False
    for segment in transcription_result["segments"]:
        assert "start" in segment
        assert "end" in segment
        assert "text" in segment
        assert "speaker" in segment # Ensure speaker key is present
        assert isinstance(segment["speaker"], str) # Ensure speaker is a string
        if segment["speaker"] != "Unknown Speaker":
            speakers_assigned = True

    # Assuming the mocked diarization assigns speakers
    assert speakers_assigned, "No speakers were assigned to any segments."

    assert "text" in transcription_result
    assert isinstance(transcription_result["text"], str)
    assert len(transcription_result["text"]) > 0

    assert "language" in transcription_result
    assert isinstance(transcription_result["language"], str)

    assert "duration" in transcription_result
    assert isinstance(transcription_result["duration"], (int, float, type(None))) # Duration can be float or None

    # Verify that the underlying WhisperX and Pyannote components were called
    mock_whisperx_components[0].load_audio.assert_called_once_with(str(audio_file))
    mock_whisperx_components[0].whisperx_model.transcribe.assert_called_once()
    mock_whisperx_components[0].align.assert_called_once()
    mock_whisperx_components[1].assert_called_once() # Check if initialize_diarization_pipeline was called
    mock_whisperx_components[1].return_value.assert_called_once_with(str(audio_file)) # Check if the pyannote pipeline instance was called


def test_transcribe_audio_file_not_found():
    """Test transcription when audio file does not exist"""
    with pytest.raises(FileNotFoundError):
        workflow_logic.transcribe_audio("test_video_id", "non_existent_file.wav")


# Removed test_transcribe_large_audio_success as the chunking logic is likely handled differently now.
# Removed tests for perform_diarization_and_assignment as it's integrated into transcribe_audio.


def test_correct_transcript_success(mock_openai_client):
    """Test successful transcript correction"""
    # Use segments with speaker info as input
    transcription_segments = [
        {"start": 0.0, "end": 1.0, "text": "this is the raw transcript text.", "speaker": "SPEAKER_00"},
        {"start": 1.5, "end": 2.5, "text": "and this is another segment.", "speaker": "SPEAKER_01"}
    ]
    correction_model = "test-correction-model"

    # Mock the LLM response for correction
    mock_chat_choice = MagicMock()
    mock_chat_choice.message.content = """SPEAKER_00: This is the corrected transcript text.
SPEAKER_01: And this is another corrected segment."""
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [mock_chat_choice]
    mock_openai_client.chat.completions.create.return_value = mock_chat_completion


    corrected = workflow_logic.correct_transcript(
        transcription_segments, correction_model
    )

    assert (
        corrected
        == """SPEAKER_00: This is the corrected transcript text.
SPEAKER_01: And this is another corrected segment."""
    )
    # Check that the prompt was constructed correctly
    call_args = (
        mock_openai_client.chat.completions.create.call_args
    )
    prompt_content = call_args.kwargs["messages"][1]["content"]  # User message
    assert "Transcript Segments (with speaker and timestamps):\n[0.00-1.00] SPEAKER_00: this is the raw transcript text.\n[1.50-2.50] SPEAKER_01: and this is another segment.\n" in prompt_content
    assert call_args.kwargs["model"] == correction_model


def test_correct_transcript_api_error(mock_openai_client):
    """Test correction failure due to API error"""
    transcription_segments = [{"start": 0.0, "end": 1.0, "text": "raw text", "speaker": "SPEAKER_00"}]
    # Simulate API error
    mock_openai_client.chat.completions.create.side_effect = (
        Exception("Correction API Error")
    )

    with pytest.raises(
        RuntimeError, match="Transcript correction API call failed with model test-correction-model: Correction API Error"
    ):
        workflow_logic.correct_transcript(transcription_segments, "test-correction-model")


def test_correct_transcript_empty_segments(mock_openai_client): # Renamed from test_correct_transcript_empty_raw
    """Test correction when transcription segments list is empty"""
    transcription_segments = []

    # Mock the LLM response for correction when input is empty
    mock_chat_choice = MagicMock()
    mock_chat_choice.message.content = "" # Expect empty output for empty input
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [mock_chat_choice]
    mock_openai_client.chat.completions.create.return_value = mock_chat_completion

    corrected = workflow_logic.correct_transcript(transcription_segments, "test-correction-model")
    assert corrected == ""
    # Verify that the OpenAI client was called
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args
    # Check that the prompt for empty segments is correct
    assert "Transcript Segments (with speaker and timestamps):\n" in call_args.kwargs["messages"][1]["content"]


def test_summarize_transcript_success(mock_openai_client):
    """Test successful transcript summarization"""
    transcript_text = "This is the full transcript text with multiple sentences."
    summarization_model = "test-summarization-model"

    # Mock the LLM response for summarization
    mock_chat_choice = MagicMock()
    mock_chat_choice.message.content = "This is the summary."
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [mock_chat_choice]
    mock_openai_client.chat.completions.create.return_value = mock_chat_completion

    summary = workflow_logic.summarize_transcript(transcript_text, summarization_model)

    assert summary == "This is the summary."
    # Check that the prompt was constructed correctly
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["messages"][1]["content"] == transcript_text
    assert call_args.kwargs["model"] == summarization_model


def test_summarize_transcript_api_error(mock_openai_client):
    """Test summarization failure due to API error"""
    transcript_text = "Some text to summarize."
    # Simulate API error
    mock_openai_client.chat.completions.create.side_effect = (
        Exception("Summarization API Error")
    )

    with pytest.raises(
        RuntimeError, match="Transcript summarization API call failed with model test-summarization-model: Summarization API Error"
    ):
        workflow_logic.summarize_transcript(transcript_text, "test-summarization-model")


def test_summarize_transcript_empty_text(mock_openai_client): # Add mock_openai_client fixture
    """Test summarization when transcript text is empty"""
    transcript_text = ""

    # Mock the LLM response for summarization when input is empty
    mock_chat_choice = MagicMock()
    mock_chat_choice.message.content = "" # Expect empty output for empty input
    mock_chat_completion = MagicMock()
    mock_chat_completion.choices = [mock_chat_choice]
    mock_openai_client.chat.completions.create.return_value = mock_chat_completion

    summary = workflow_logic.summarize_transcript(transcript_text, "test-summarization-model")
    assert summary == ""
    # Verify that the OpenAI client was called
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args.kwargs["messages"][1]["content"] == transcript_text


def test_generate_html_reader(tmp_path):
    """Test HTML reader generation"""
    corrected_transcript = "SPEAKER_01: This is the corrected transcript."
    output_path = tmp_path / "reader.html"

    workflow_logic.generate_html_reader(corrected_transcript, str(output_path))
    
    assert os.path.exists(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Transcript Reader" in content
    assert corrected_transcript in content
    assert "<pre>" in content
    assert "</pre>" in content


def test_generate_html_reader_creates_directory(tmp_path):
    """Test HTML reader generation creates output directory if needed"""
    corrected_transcript = "Some text."
    output_dir = tmp_path / "new_dir"
    output_path = output_dir / "reader.html"

    workflow_logic.generate_html_reader(corrected_transcript, str(output_path))

    assert os.path.exists(output_path)
    assert os.path.isdir(output_dir)


# --- END OF FILE tests/test_workflow.py ---

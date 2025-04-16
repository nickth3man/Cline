# --- START OF FILE tests/test_workflow.py ---

import pytest
import os
import subprocess
from unittest.mock import patch, MagicMock, mock_open


# Mock environment variables before importing the module under test
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-fakekey")
    monkeypatch.setenv("HF_TOKEN", "hf_faketoken")
    # Import the module *after* setting env vars
    global workflow_logic
    import workflow_logic

    # Reload to ensure mocks are applied if already imported
    import importlib

    importlib.reload(workflow_logic)


# --- Fixtures ---


@pytest.fixture
def sample_diarization_result():
    return [
        {"speaker": "SPEAKER_00", "start": 0.500, "end": 2.800},
        {"speaker": "SPEAKER_01", "start": 3.100, "end": 5.200},
    ]


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client used in workflow_logic"""
    with patch("workflow_logic.client", autospec=True) as mock_client:
        # Mock the transcription response
        mock_transcription = MagicMock()
        mock_transcription.text = "This is the raw transcript text."
        mock_client.with_options.return_value.audio.transcriptions.create.return_value = (
            mock_transcription
        )

        # Mock the chat completion (correction) response
        mock_choice = MagicMock()
        mock_choice.message.content = "SPEAKER_00: This is the corrected transcript.\nSPEAKER_01: With speaker labels."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.with_options.return_value.chat.completions.create.return_value = (
            mock_completion
        )

        yield mock_client


@pytest.fixture
def mock_pyannote_pipeline():
    """Fixture to mock the pyannote pipeline"""
    # Mock the pipeline object loading and calling
    mock_pipeline_instance = MagicMock()

    # Define the mock output for pipeline call
    mock_turn_1 = MagicMock(start=0.5, end=2.8)
    mock_turn_2 = MagicMock(start=3.1, end=5.2)
    mock_track_1 = (mock_turn_1, "_", "SPEAKER_00")
    mock_track_2 = (mock_turn_2, "_", "SPEAKER_01")
    mock_diarization_result = MagicMock()
    mock_diarization_result.itertracks.return_value = iter([mock_track_1, mock_track_2])
    mock_diarization_result.labels.return_value = [
        "SPEAKER_00",
        "SPEAKER_01",
    ]  # Needed for log message

    mock_pipeline_instance.return_value = (
        mock_diarization_result  # Mock the __call__ method
    )

    # Patch the Pipeline.from_pretrained method AND the pipeline object in the module
    with patch(
        "pyannote.audio.Pipeline.from_pretrained", return_value=mock_pipeline_instance
    ) as mock_from_pretrained, patch(
        "workflow_logic.diarization_pipeline", mock_pipeline_instance
    ) as mock_pipeline_obj:
        # Ensure the reloaded module uses the mocked instance
        workflow_logic.diarization_pipeline = mock_pipeline_instance
        yield mock_pipeline_instance


@pytest.fixture
def mock_ffmpeg():
    """Fixture to mock ffmpeg subprocess calls"""
    with patch("subprocess.run") as mock_run:
        # Mock the successful return for version check and conversion
        mock_process = MagicMock()
        mock_process.stderr = b""
        mock_process.stdout = b"ffmpeg version ..."
        mock_run.return_value = mock_process
        # Also patch the internal check flags for consistency
        with patch("workflow_logic._ffmpeg_checked", True), patch(
            "workflow_logic._ffmpeg_present", True
        ):
            yield mock_run


# --- Test Cases ---


def test_check_ffmpeg_found(mock_ffmpeg):
    """Test check_ffmpeg when ffmpeg is found"""
    assert workflow_logic.check_ffmpeg() is True
    mock_ffmpeg.assert_called_with(
        ["ffmpeg", "-version"], check=True, capture_output=True
    )


def test_check_ffmpeg_not_found():
    """Test check_ffmpeg when ffmpeg is not found"""
    with patch("subprocess.run", side_effect=FileNotFoundError) as mock_run:
        # Reset checked flags for this specific test
        workflow_logic._ffmpeg_checked = False
        assert workflow_logic.check_ffmpeg() is False
        workflow_logic._ffmpeg_checked = False  # Reset again after test
        mock_run.assert_called_with(
            ["ffmpeg", "-version"], check=True, capture_output=True
        )


def test_extract_or_convert_audio_success(mock_ffmpeg, tmp_path):
    """Test successful audio conversion"""
    input_file = tmp_path / "input.mp4"
    output_file = tmp_path / "output.wav"
    input_file.touch()  # Create dummy input file

    result_path = workflow_logic.extract_or_convert_audio(
        str(input_file), str(output_file)
    )

    assert result_path == str(output_file)
    mock_ffmpeg.assert_any_call(
        ["ffmpeg", "-version"], check=True, capture_output=True
    )  # Check version call
    mock_ffmpeg.assert_called_with(  # Check conversion call
        [
            "ffmpeg",
            "-i",
            str(input_file),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-nostdin",
            "-y",
            str(output_file),
        ],
        check=True,
        capture_output=True,
    )


def test_extract_or_convert_audio_ffmpeg_fail(tmp_path):
    """Test audio conversion when ffmpeg command fails"""
    input_file = tmp_path / "input.mp4"
    output_file = tmp_path / "output.wav"
    input_file.touch()

    # Mock ffmpeg failure
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "cmd", stderr=b"ffmpeg error"),
    ) as mock_run:
        # Ensure ffmpeg check initially passes
        with patch("workflow_logic.check_ffmpeg", return_value=True):
            with pytest.raises(RuntimeError, match="FFmpeg failed"):
                workflow_logic.extract_or_convert_audio(
                    str(input_file), str(output_file)
                )


def test_extract_or_convert_audio_ffmpeg_not_present(tmp_path):
    """Test audio conversion when ffmpeg is not installed"""
    input_file = tmp_path / "input.mp4"
    output_file = tmp_path / "output.wav"
    input_file.touch()

    with patch("workflow_logic.check_ffmpeg", return_value=False):
        # Reset internal flags for this test
        workflow_logic._ffmpeg_checked = False
        with pytest.raises(RuntimeError, match="FFmpeg is required"):
            workflow_logic.extract_or_convert_audio(str(input_file), str(output_file))
        workflow_logic._ffmpeg_checked = False  # Reset after test


def test_transcribe_audio_success(mock_openai_client, tmp_path):
    """Test successful transcription"""
    audio_file = tmp_path / "audio.wav"
    # Use mock_open to simulate reading the file
    with patch("builtins.open", mock_open(read_data=b"dummy_audio_data")) as mock_file:
        transcript = workflow_logic.transcribe_audio(str(audio_file))

    assert transcript == "This is the raw transcript text."
    mock_file.assert_called_once_with(str(audio_file), "rb")
    mock_openai_client.with_options.return_value.audio.transcriptions.create.assert_called_once()


def test_transcribe_audio_api_error(mock_openai_client, tmp_path):
    """Test transcription failure due to API error"""
    audio_file = tmp_path / "audio.wav"
    # Simulate API error
    mock_openai_client.with_options.return_value.audio.transcriptions.create.side_effect = Exception(
        "API Connection Error"
    )

    with patch("builtins.open", mock_open(read_data=b"dummy_audio_data")):
        with pytest.raises(
            RuntimeError, match="Transcription failed: API Connection Error"
        ):
            workflow_logic.transcribe_audio(str(audio_file))


def test_transcribe_audio_file_not_found(mock_openai_client):
    """Test transcription when audio file does not exist"""
    with pytest.raises(FileNotFoundError):
        workflow_logic.transcribe_audio("non_existent_file.wav")


def test_diarize_speakers_success(mock_pyannote_pipeline, tmp_path):
    """Test successful diarization"""
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()  # Create dummy file

    segments = workflow_logic.diarize_speakers(str(audio_file))

    assert len(segments) == 2
    assert segments[0] == {"speaker": "SPEAKER_00", "start": 0.500, "end": 2.800}
    assert segments[1] == {"speaker": "SPEAKER_01", "start": 3.100, "end": 5.200}
    mock_pyannote_pipeline.assert_called_once_with(str(audio_file), num_speakers=None)


def test_diarize_speakers_pipeline_fail(mock_pyannote_pipeline, tmp_path):
    """Test diarization when pipeline itself raises an error"""
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    # Simulate pipeline call failure
    mock_pyannote_pipeline.side_effect = Exception("Pipeline error")

    # Should log error and return empty list, not raise exception
    segments = workflow_logic.diarize_speakers(str(audio_file))
    assert segments == []


def test_diarize_speakers_pipeline_not_loaded(tmp_path):
    """Test diarization when pipeline object is None"""
    audio_file = tmp_path / "audio.wav"
    audio_file.touch()
    # Ensure pipeline is None for this test
    with patch("workflow_logic.diarization_pipeline", None):
        segments = workflow_logic.diarize_speakers(str(audio_file))
        assert segments == []


def test_format_diarization_for_llm(sample_diarization_result):
    """Test formatting of diarization results"""
    formatted = workflow_logic.format_diarization_for_llm(sample_diarization_result)
    assert "Speaker Turns" in formatted
    assert "SPEAKER_00 0:00:00.500-0:00:02.800" in formatted
    assert "SPEAKER_01 0:00:03.100-0:00:05.200" in formatted


def test_format_diarization_for_llm_empty():
    """Test formatting when diarization result is empty"""
    formatted = workflow_logic.format_diarization_for_llm([])
    assert "Speaker diarization information is not available." in formatted


def test_correct_transcript_success(mock_openai_client, sample_diarization_result):
    """Test successful transcript correction"""
    raw = "this is the raw transcript text."
    correction_model = "test-correction-model"

    corrected = workflow_logic.correct_transcript(
        raw, sample_diarization_result, correction_model
    )

    assert (
        corrected
        == "SPEAKER_00: This is the corrected transcript.\nSPEAKER_01: With speaker labels."
    )
    # Check that the prompt was constructed correctly (simplified check)
    call_args = (
        mock_openai_client.with_options.return_value.chat.completions.create.call_args
    )
    prompt_content = call_args.kwargs["messages"][1]["content"]  # User message
    assert raw in prompt_content
    assert (
        "SPEAKER_00 0:00:00.500-0:00:02.800" in prompt_content
    )  # Check formatted diarization
    assert call_args.kwargs["model"] == correction_model


def test_correct_transcript_api_error(mock_openai_client, sample_diarization_result):
    """Test correction failure due to API error"""
    raw = "raw text"
    # Simulate API error
    mock_openai_client.with_options.return_value.chat.completions.create.side_effect = (
        Exception("Correction API Error")
    )

    with pytest.raises(
        RuntimeError, match="Transcript correction failed: Correction API Error"
    ):
        workflow_logic.correct_transcript(raw, sample_diarization_result)


def test_correct_transcript_empty_raw():
    """Test correction when raw transcript is empty"""
    corrected = workflow_logic.correct_transcript("", [])
    assert corrected == ""


# --- END OF FILE tests/test_workflow.py ---

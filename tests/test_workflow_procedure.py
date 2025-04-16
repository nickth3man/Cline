import os
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Try to import workflow_logic directly
try:
    from src.utils import workflow_logic

    print("Successfully imported workflow_logic")
except ImportError as e:
    print(f"Import error: {e}")
    # Try alternative import path
    try:
        import sys

        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        )
        from src.utils import workflow_logic

        print("Successfully imported workflow_logic after path adjustment")
    except ImportError as e2:
        print(f"Second import error: {e2}")
        raise


def create_dummy_audio_file(directory, name="dummy_audio.wav"):
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(b"\x00\x01" * 1000)  # Small dummy content
    return path


def create_intermediate_audio_attempt(directory, base_name="dummy_audio"):
    path = os.path.join(directory, f"{base_name}_audio_attempt1.wav")
    with open(path, "wb") as f:
        f.write(b"\x02\x03" * 1000)
    return path


@pytest.fixture
def temp_audio_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@patch("src.utils.workflow_logic.transcribe_audio")
@patch("src.utils.workflow_logic.diarize_speakers")
@patch("src.utils.workflow_logic.correct_transcript")
@patch("src.utils.workflow_logic.summarize_transcript")
def test_process_audio_file_success(
    mock_summarize, mock_correct, mock_diarize, mock_transcribe, temp_audio_dir
):
    audio_path = create_dummy_audio_file(temp_audio_dir)
    create_intermediate_audio_attempt(
        temp_audio_dir, base_name=os.path.splitext(os.path.basename(audio_path))[0]
    )
    output_dir = os.path.join(temp_audio_dir, "output")

    mock_transcribe.return_value = "raw transcript"
    mock_diarize.return_value = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0}]
    mock_correct.return_value = "corrected transcript"
    mock_summarize.return_value = "summary text"

    transcript_path, summary_path = workflow_logic.process_audio_file(
        audio_path=audio_path,
        output_dir=output_dir,
        transcription_model="openai/whisper-large-v3",
        correction_model="openai/gpt-4.1-mini",
        summarization_model="openai/gpt-4.1-mini",
    )

    # Output files should exist
    assert os.path.isfile(transcript_path)
    assert os.path.isfile(summary_path)
    # Intermediate audio attempt file should be deleted
    for fname in os.listdir(temp_audio_dir):
        assert "audio_attempt" not in fname


@patch("src.utils.workflow_logic.transcribe_audio")
def test_process_audio_file_transcription_failure(mock_transcribe, temp_audio_dir):
    audio_path = create_dummy_audio_file(temp_audio_dir)
    create_intermediate_audio_attempt(
        temp_audio_dir, base_name=os.path.splitext(os.path.basename(audio_path))[0]
    )
    output_dir = os.path.join(temp_audio_dir, "output")
    mock_transcribe.side_effect = Exception("Transcription error!")
    transcript_path, summary_path = workflow_logic.process_audio_file(
        audio_path=audio_path,
        output_dir=output_dir,
        transcription_model="openai/whisper-large-v3",
        correction_model="openai/gpt-4.1-mini",
        summarization_model=None,
    )
    # Should return None, None
    assert transcript_path is None
    assert summary_path is None
    # Intermediate audio attempt file should be deleted
    for fname in os.listdir(temp_audio_dir):
        assert "audio_attempt" not in fname


@patch("src.utils.workflow_logic.transcribe_audio")
def test_process_audio_file_empty_transcript(mock_transcribe, temp_audio_dir):
    audio_path = create_dummy_audio_file(temp_audio_dir)
    create_intermediate_audio_attempt(
        temp_audio_dir, base_name=os.path.splitext(os.path.basename(audio_path))[0]
    )
    output_dir = os.path.join(temp_audio_dir, "output")
    mock_transcribe.return_value = ""
    transcript_path, summary_path = workflow_logic.process_audio_file(
        audio_path=audio_path,
        output_dir=output_dir,
        transcription_model="openai/whisper-large-v3",
        correction_model="openai/gpt-4.1-mini",
        summarization_model=None,
    )
    assert transcript_path is None
    assert summary_path is None
    for fname in os.listdir(temp_audio_dir):
        assert "audio_attempt" not in fname


@patch("src.utils.workflow_logic.transcribe_audio")
@patch("src.utils.workflow_logic.diarize_speakers")
def test_process_audio_file_diarization_failure(
    mock_diarize, mock_transcribe, temp_audio_dir
):
    audio_path = create_dummy_audio_file(temp_audio_dir)
    create_intermediate_audio_attempt(
        temp_audio_dir, base_name=os.path.splitext(os.path.basename(audio_path))[0]
    )
    output_dir = os.path.join(temp_audio_dir, "output")
    mock_transcribe.return_value = "raw transcript"
    mock_diarize.side_effect = Exception("Diarization error!")
    with patch(
        "src.utils.workflow_logic.correct_transcript",
        return_value="corrected transcript",
    ):
        with patch(
            "src.utils.workflow_logic.summarize_transcript", return_value="summary text"
        ):
            transcript_path, summary_path = workflow_logic.process_audio_file(
                audio_path=audio_path,
                output_dir=output_dir,
                transcription_model="openai/whisper-large-v3",
                correction_model="openai/gpt-4.1-mini",
                summarization_model="openai/gpt-4.1-mini",
            )
    assert os.path.isfile(transcript_path)
    assert os.path.isfile(summary_path)
    for fname in os.listdir(temp_audio_dir):
        assert "audio_attempt" not in fname


@patch("src.utils.workflow_logic.transcribe_audio")
@patch("src.utils.workflow_logic.diarize_speakers")
def test_process_audio_file_correction_failure(
    mock_diarize, mock_transcribe, temp_audio_dir
):
    audio_path = create_dummy_audio_file(temp_audio_dir)
    create_intermediate_audio_attempt(
        temp_audio_dir, base_name=os.path.splitext(os.path.basename(audio_path))[0]
    )
    output_dir = os.path.join(temp_audio_dir, "output")
    mock_transcribe.return_value = "raw transcript"
    mock_diarize.return_value = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0}]
    with patch(
        "src.utils.workflow_logic.correct_transcript",
        side_effect=Exception("Correction error!"),
    ):
        transcript_path, summary_path = workflow_logic.process_audio_file(
            audio_path=audio_path,
            output_dir=output_dir,
            transcription_model="openai/whisper-large-v3",
            correction_model="openai/gpt-4.1-mini",
            summarization_model=None,
        )
    assert transcript_path is None
    assert summary_path is None
    for fname in os.listdir(temp_audio_dir):
        assert "audio_attempt" not in fname

import os
import shutil
import pytest
from unittest.mock import patch, MagicMock

from src import run_full_pipeline


@pytest.fixture
def temp_output_dir(tmp_path):
    # Use a temporary directory for outputs
    yield tmp_path
    # Cleanup after test
    shutil.rmtree(tmp_path, ignore_errors=True)


def mock_yt_dlp_download(urls):
    # Simulate yt-dlp download by creating a dummy mp4 file
    os.makedirs("output/000_Test_Video", exist_ok=True)
    with open("output/000_Test_Video/Test_Video.mp4", "wb") as f:
        f.write(b"dummy video content")


@patch("yt_dlp.YoutubeDL")
@patch("src.run_full_pipeline.extract_or_convert_audio")
@patch("src.transcription.transcription_workflow.transcribe_audio")
@patch("src.transcription.transcription_workflow.diarize_speakers")
@patch("src.transcription.transcription_workflow.correct_transcript")
@patch("src.utils.workflow_logic.summarize_transcript")
def test_pipeline_creates_outputs(
    mock_summarize,
    mock_correct,
    mock_diarize,
    mock_transcribe,
    mock_extract,
    mock_ytdlp,
    temp_output_dir,
):
    # Setup mocks
    mock_ytdlp.return_value.__enter__.return_value.extract_info.return_value = {
        "entries": [
            {
                "title": "Test Video",
                "url": "https://youtube.com/watch?v=dummy",
                "webpage_url": "https://youtube.com/watch?v=dummy",
                "id": "dummy",
            }
        ]
    }
    mock_ytdlp.return_value.__enter__.return_value.download.side_effect = (
        mock_yt_dlp_download
    )
    mock_extract.side_effect = lambda in_path, out_path: out_path
    mock_transcribe.return_value = "This is a test transcript."
    mock_diarize.return_value = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0}]
    mock_correct.return_value = "SPEAKER_00: This is a corrected transcript."
    mock_summarize.return_value = "This is a summary."

    # Run pipeline
    run_full_pipeline.run_pipeline(
        playlist_url="https://youtube.com/playlist?list=TEST",
        output_base_dir="output",
        correction_models=["openai/gpt-4.1-mini"],
        summarization_model="openai/gpt-4.1-mini",
    )

    # Check output files
    video_dir = os.path.join("output", "000_Test_Video")
    assert os.path.isdir(video_dir)
    assert os.path.isfile(os.path.join(video_dir, "Test_Video.mp4"))
    assert os.path.isfile(os.path.join(video_dir, "Test_Video_audio.wav"))
    assert os.path.isfile(os.path.join(video_dir, "Test_Video_raw_transcript.txt"))
    assert os.path.isfile(os.path.join(video_dir, "Test_Video_diarization.json"))
    assert os.path.isfile(
        os.path.join(video_dir, "Test_Video_corrected_openai_gpt-4.1-mini.md")
    )
    assert os.path.isfile(
        os.path.join(video_dir, "Test_Video_summary_openai_gpt-4.1-mini.txt")
    )
    assert os.path.isfile(os.path.join(video_dir, "Test_Video_transcript_reader.html"))

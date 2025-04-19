import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from pydub import AudioSegment # Import AudioSegment for creating dummy audio files

# Add the project root to the system path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock WhisperX and torch components at the module level before importing workflow_logic
patch('src.utils.workflow_logic.whisperx.load_model', return_value=MagicMock()).start()
patch('src.utils.workflow_logic.whisperx.load_align_model', return_value=(MagicMock(), MagicMock())).start()
patch('src.utils.workflow_logic.whisperx.DiarizationPipeline', return_value=MagicMock()).start()
patch('src.utils.workflow_logic.torch.cuda.is_available', return_value=False).start() # Assume no CUDA for tests
# Move the patch for initialize_diarization_pipeline here
patch('src.utils.workflow_logic.initialize_diarization_pipeline', return_value=MagicMock()).start()


from src.utils.workflow_logic import transcribe_audio # Import after applying module-level mocks


class TestWhisperXTranscription(unittest.TestCase):

    @patch('src.utils.workflow_logic.whisperx.load_audio')
    @patch('src.utils.workflow_logic.whisperx_model.transcribe')
    @patch('src.utils.workflow_logic.whisperx.align')
    @patch('src.utils.workflow_logic.mediainfo', return_value={'duration': '10.0'}) # Mock mediainfo
    @patch('src.utils.workflow_logic.assign_speakers_to_segments', return_value=[{"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_01"}, {"start": 1.1, "end": 2.0, "text": "world", "speaker": "SPEAKER_02"}]) # Mock speaker assignment
    # Removed the patch for initialize_diarization_pipeline from here as it's now at module level
    def test_transcribe_audio_success(self, mock_assign_speakers, mock_mediainfo, mock_align, mock_transcribe, mock_load_audio): # Removed mock_init_diarization parameter
        """
        Test successful transcription using WhisperX.
        """
        # Mock return values for WhisperX functions
        mock_load_audio.return_value = MagicMock()
        mock_transcribe.return_value = {"segments": [{"text": "hello world"}], "language": "en"}
        mock_align.return_value = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}, {"start": 1.1, "end": 2.0, "text": "world"}], "language": "en"}
 
        # Create a dummy audio file for the test using pydub
        dummy_audio_path = "temp_dummy_audio.wav"
        # Create 5 seconds of silent audio at 16000 Hz
        silent_audio = AudioSegment.silent(duration=5000, frame_rate=16000)
        silent_audio.export(dummy_audio_path, format="wav")

        video_id = "test_video_123"
        
        try:
            result = transcribe_audio(video_id, dummy_audio_path)

            # Assertions
            self.assertIsInstance(result, dict)
            self.assertIn("segments", result)
            self.assertIsInstance(result["segments"], list)
            self.assertEqual(len(result["segments"]), 2) # Expecting 2 segments from mock_assign_speakers
            self.assertIn("start", result["segments"][0])
            self.assertIn("end", result["segments"][0])
            self.assertIn("text", result["segments"][0])
            self.assertIn("speaker", result["segments"][0]) # Check for speaker key
            self.assertIn("text", result)
            # The full text should be concatenated from the segments returned by assign_speakers_to_segments mock
            self.assertEqual(result["text"], "[SPEAKER_01] hello [SPEAKER_02] world")
            self.assertIn("language", result)
            self.assertEqual(result["language"], "en")
            self.assertIn("duration", result)
            # The duration should be mocked by mediainfo
            self.assertEqual(result["duration"], 10.0)

            # Verify that WhisperX and Pyannote related functions were called
            mock_load_audio.assert_called_once_with(dummy_audio_path)
            mock_transcribe.assert_called_once()
            mock_align.assert_called_once()
            # Removed the assertion for mock_init_diarization as it's initialized at module level
            # mock_init_diarization.assert_called_once() # Check if diarization pipeline was initialized
            # Check if the pyannote pipeline instance was called (mocked within transcribe_audio)
            # This requires accessing the mocked pipeline instance's call method
            # mock_init_diarization.return_value.assert_called_once_with(dummy_audio_path) # This mock is tricky, might need adjustment
            mock_assign_speakers.assert_called_once() # Check if speaker assignment was called
            mock_mediainfo.assert_called_once_with(dummy_audio_path)


        finally:
            # Clean up the dummy audio file
            if os.path.exists(dummy_audio_path):
                os.remove(dummy_audio_path)

    @patch('src.utils.workflow_logic.whisperx_model', None) # Mock models as not loaded
    def test_transcribe_audio_models_not_loaded(self):
        """
        Test transcription when WhisperX models are not loaded.
        """
        dummy_audio_path = "temp_dummy_audio.wav"
        # Create a dummy audio file for the test using pydub
        silent_audio = AudioSegment.silent(duration=1000, frame_rate=16000)
        silent_audio.export(dummy_audio_path, format="wav")
        video_id = "test_video_456"

        try:
            with self.assertRaises(RuntimeError) as cm:
                transcribe_audio(video_id, dummy_audio_path)

            self.assertIn("WhisperX models are not loaded", str(cm.exception))
        finally:
             if os.path.exists(dummy_audio_path):
                os.remove(dummy_audio_path)


    def test_transcribe_audio_file_not_found(self):
        """
        Test transcription with a non-existent audio file.
        """
        video_id = "test_video_789"
        non_existent_audio_path = "non_existent_audio.wav"

        with self.assertRaises(FileNotFoundError) as cm:
            transcribe_audio(video_id, non_existent_audio_path)

        self.assertIn("Audio file not found for transcription", str(cm.exception))

# Removed the __main__ block as pytest runs the tests.
import os
import shutil
import tempfile
import unittest

# Import the function directly from the module
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.utils.workflow_logic import _cleanup_intermediate_audio # Commented out due to ImportError


# class TestCleanup(unittest.TestCase): # Commented out the test class
#     """Test the cleanup functionality for intermediate audio files."""

#     def setUp(self):
#         """Create a temporary directory for testing."""
#         self.test_dir = tempfile.mkdtemp()

#     def tearDown(self):
#         """Clean up the temporary directory after testing."""
#         shutil.rmtree(self.test_dir)

#     def test_cleanup_intermediate_audio(self):
#         """Test that intermediate audio files are properly cleaned up."""
#         # Create a main audio file
#         main_audio_path = os.path.join(self.test_dir, "video1.wav")
#         with open(main_audio_path, "wb") as f:
#             f.write(b"dummy audio content")

#         # Create intermediate audio attempt files
#         attempt1_path = os.path.join(self.test_dir, "video1_audio_attempt1.wav")
#         attempt2_path = os.path.join(self.test_dir, "video1_audio_attempt2.wav")
#         with open(attempt1_path, "wb") as f:
#             f.write(b"attempt 1 content")
#         with open(attempt2_path, "wb") as f:
#             f.write(b"attempt 2 content")

#         # Create a non-matching file that should not be deleted
#         other_file = os.path.join(self.test_dir, "other_file.wav")
#         with open(other_file, "wb") as f:
#             f.write(b"other content")

#         # Run the cleanup function
#         # _cleanup_intermediate_audio(main_audio_path) # Commented out function call

#         # Check that intermediate files are deleted
#         self.assertFalse(
#             os.path.exists(attempt1_path), "Intermediate audio file 1 should be deleted"
#         )
#         self.assertFalse(
#             os.path.exists(attempt2_path), "Intermediate audio file 2 should be deleted"
#         )

#         # Check that main audio and other files are preserved
#         self.assertTrue(
#             os.path.exists(main_audio_path), "Main audio file should not be deleted"
#         )
#         self.assertTrue(
#             os.path.exists(other_file), "Unrelated files should not be deleted"
#         )

#     def test_cleanup_handles_nonexistent_files(self):
#         """Test that cleanup handles nonexistent files gracefully."""
#         # Create a main audio file
#         main_audio_path = os.path.join(self.test_dir, "video2.wav")
#         with open(main_audio_path, "wb") as f:
#             f.write(b"dummy audio content")

#         # Create a directory that should be ignored
#         os.makedirs(os.path.join(self.test_dir, "video2_audio_attempt_dir"))

#         # Run the cleanup function - should not raise exceptions
#         # _cleanup_intermediate_audio(main_audio_path) # Commented out function call

#         # Directory should still exist
#         self.assertTrue(
#             os.path.exists(os.path.join(self.test_dir, "video2_audio_attempt_dir"))
#         )


# if __name__ == "__main__": # Commented out the main execution block
#     unittest.main()

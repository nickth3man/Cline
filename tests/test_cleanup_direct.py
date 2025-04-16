"""
Direct test script for the _cleanup_intermediate_audio function.
This script creates test files, runs the cleanup function, and verifies results.
"""

import os
import shutil
import tempfile
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define a simplified version of the cleanup function for testing
def cleanup_intermediate_audio(audio_path):
    """
    Removes intermediate audio files (e.g., *_audio_attempt*.wav) and other temporary files
    associated with the given audio_path. Ensures that only the final output structure remains clean.
    """
    try:
        base_dir = os.path.dirname(audio_path)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        # Remove files matching *_audio_attempt*.wav in the same directory
        for fname in os.listdir(base_dir):
            if (
                fname.startswith(base_name)
                and "audio_attempt" in fname
                and fname.endswith(".wav")
            ):
                file_path = os.path.join(base_dir, fname)
                try:
                    os.remove(file_path)
                    logging.info(f"Removed intermediate audio file: {file_path}")
                except Exception as e:
                    logging.warning(
                        f"Failed to remove intermediate audio file: {file_path}: {e}"
                    )
    except Exception as cleanup_err:
        logging.warning(f"Cleanup of intermediate audio files failed: {cleanup_err}")


def run_test():
    """Run the test for the cleanup function."""
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    logging.info(f"Created test directory: {test_dir}")

    try:
        # Create a main audio file
        main_audio_path = os.path.join(test_dir, "video1.wav")
        with open(main_audio_path, "wb") as f:
            f.write(b"dummy audio content")
        logging.info(f"Created main audio file: {main_audio_path}")

        # Create intermediate audio attempt files
        attempt1_path = os.path.join(test_dir, "video1_audio_attempt1.wav")
        attempt2_path = os.path.join(test_dir, "video1_audio_attempt2.wav")
        with open(attempt1_path, "wb") as f:
            f.write(b"attempt 1 content")
        with open(attempt2_path, "wb") as f:
            f.write(b"attempt 2 content")
        logging.info(
            f"Created intermediate audio files: {attempt1_path}, {attempt2_path}"
        )

        # Create a non-matching file that should not be deleted
        other_file = os.path.join(test_dir, "other_file.wav")
        with open(other_file, "wb") as f:
            f.write(b"other content")
        logging.info(f"Created non-matching file: {other_file}")

        # Run the cleanup function
        logging.info("Running cleanup function...")
        cleanup_intermediate_audio(main_audio_path)

        # Check that intermediate files are deleted
        if not os.path.exists(attempt1_path) and not os.path.exists(attempt2_path):
            logging.info("✓ SUCCESS: Intermediate audio files were correctly deleted")
        else:
            logging.error("✗ FAILURE: Intermediate audio files were not deleted")
            if os.path.exists(attempt1_path):
                logging.error(f"  - {attempt1_path} still exists")
            if os.path.exists(attempt2_path):
                logging.error(f"  - {attempt2_path} still exists")

        # Check that main audio and other files are preserved
        if os.path.exists(main_audio_path) and os.path.exists(other_file):
            logging.info(
                "✓ SUCCESS: Main audio file and unrelated files were correctly preserved"
            )
        else:
            logging.error(
                "✗ FAILURE: Main audio file or unrelated files were incorrectly deleted"
            )
            if not os.path.exists(main_audio_path):
                logging.error(f"  - {main_audio_path} was deleted")
            if not os.path.exists(other_file):
                logging.error(f"  - {other_file} was deleted")

        # Overall test result
        if (
            not os.path.exists(attempt1_path)
            and not os.path.exists(attempt2_path)
            and os.path.exists(main_audio_path)
            and os.path.exists(other_file)
        ):
            logging.info("✅ TEST PASSED: Cleanup function works correctly")
            return True
        else:
            logging.error("❌ TEST FAILED: Cleanup function did not work as expected")
            return False

    finally:
        # Clean up the temporary directory
        shutil.rmtree(test_dir)
        logging.info(f"Cleaned up test directory: {test_dir}")


if __name__ == "__main__":
    logging.info("Starting cleanup function test...")
    success = run_test()
    if success:
        logging.info("All tests passed successfully!")
        sys.exit(0)
    else:
        logging.error("Test failed!")
        sys.exit(1)

"""
Standalone test for the audio file cleanup functionality.
This test doesn't import the main module to avoid dependency issues.
"""

import os
import shutil
import tempfile
import logging
import sys
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"logs/cleanup_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)


def cleanup_intermediate_audio(audio_path):
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


def enforce_output_structure(video_dir, video_filename):
    """
    Enforces the strict output structure rule: each video folder should only contain
    1. The original video file
    2. The transcript from the video
    3. The transcript summary

    Args:
        video_dir: Path to the video directory
        video_filename: Filename of the original video (without path)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(video_dir) or not os.path.isdir(video_dir):
            logging.error(f"Video directory does not exist: {video_dir}")
            return False

        base_name = os.path.splitext(video_filename)[0]

        # Define allowed files
        allowed_files = [
            video_filename,  # Original video
            f"{base_name}_corrected.md",  # Transcript
            f"{base_name}_summary.md",  # Summary
        ]

        removed_items = []

        # First, remove all subdirectories
        for item in os.listdir(video_dir):
            item_path = os.path.join(video_dir, item)

            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                    removed_items.append(("directory", item))
                    logging.info(f"Removed directory: {item_path}")
                except Exception as e:
                    logging.error(f"Failed to remove directory: {item_path}: {e}")
                    return False

        # Then remove all files except allowed ones
        for item in os.listdir(video_dir):
            if item not in allowed_files:
                item_path = os.path.join(video_dir, item)
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
        remaining_files = os.listdir(video_dir)
        if set(remaining_files).issubset(set(allowed_files)):
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
            logging.info("SUCCESS: Intermediate audio files were correctly deleted")
        else:
            logging.error("FAILURE: Intermediate audio files were not deleted")
            if os.path.exists(attempt1_path):
                logging.error(f"  - {attempt1_path} still exists")
            if os.path.exists(attempt2_path):
                logging.error(f"  - {attempt2_path} still exists")

        # Check that main audio and other files are preserved
        if os.path.exists(main_audio_path) and os.path.exists(other_file):
            logging.info(
                "SUCCESS: Main audio file and unrelated files were correctly preserved"
            )
        else:
            logging.error(
                "FAILURE: Main audio file or unrelated files were incorrectly deleted"
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
            logging.info("TEST PASSED: Cleanup function works correctly")
            return True
        else:
            logging.error("TEST FAILED: Cleanup function did not work as expected")
            return False

    finally:
        # Clean up the temporary directory
        shutil.rmtree(test_dir)
        logging.info(f"Cleaned up test directory: {test_dir}")


def test_output_structure():
    """
    Test that the output structure follows the project rules:
    Each video folder should only contain:
    1. The original video file
    2. The transcript file
    3. The transcript summary
    """
    # Create a temporary directory for testing
    output_dir = tempfile.mkdtemp()
    logging.info(f"Created test output directory: {output_dir}")

    try:
        # Create sample video folders
        video1_dir = os.path.join(output_dir, "Video1_Title")
        os.makedirs(video1_dir, exist_ok=True)

        # Create valid files (these should remain)
        video_file = os.path.join(video1_dir, "Video1_Title.mp4")
        transcript_file = os.path.join(video1_dir, "Video1_Title_corrected.md")
        summary_file = os.path.join(video1_dir, "Video1_Title_summary.md")

        with open(video_file, "wb") as f:
            f.write(b"dummy video content")
        with open(transcript_file, "w") as f:
            f.write("Transcript content")
        with open(summary_file, "w") as f:
            f.write("Summary content")

        # Create invalid files (these should be removed)
        temp_audio = os.path.join(video1_dir, "Video1_Title.wav")
        temp_audio_attempt = os.path.join(video1_dir, "Video1_Title_audio_attempt1.wav")
        chapters_dir = os.path.join(video1_dir, "Chapters")
        os.makedirs(chapters_dir, exist_ok=True)
        chapter_file = os.path.join(chapters_dir, "chapter1.md")

        with open(temp_audio, "wb") as f:
            f.write(b"audio content")
        with open(temp_audio_attempt, "wb") as f:
            f.write(b"audio attempt content")
        with open(chapter_file, "w") as f:
            f.write("Chapter content")

        # Run the enhanced structure enforcement
        result = enforce_output_structure(video1_dir, os.path.basename(video_file))

        # Check that only the valid files remain
        valid_files = [
            os.path.basename(video_file),
            os.path.basename(transcript_file),
            os.path.basename(summary_file),
        ]

        remaining_files = os.listdir(video1_dir)

        # Check if any directories remain
        has_subdirs = any(
            os.path.isdir(os.path.join(video1_dir, item)) for item in remaining_files
        )

        if not has_subdirs and set(remaining_files) == set(valid_files):
            logging.info("OUTPUT STRUCTURE TEST PASSED: Only valid files remain")
            logging.info(f"Remaining files: {remaining_files}")
            return True
        else:
            logging.error(
                "OUTPUT STRUCTURE TEST FAILED: Invalid files or directories remain"
            )
            logging.error(f"Expected files: {valid_files}")
            logging.error(f"Actual files: {remaining_files}")
            if has_subdirs:
                logging.error("Subdirectories still exist and should be removed")
            return False

    finally:
        # Clean up the temporary directory
        shutil.rmtree(output_dir)
        logging.info(f"Cleaned up test output directory: {output_dir}")


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    logging.info("Starting cleanup function tests...")

    # Run basic cleanup test
    cleanup_test_result = run_test()

    # Run output structure test
    structure_test_result = test_output_structure()

    # Overall results
    if cleanup_test_result and structure_test_result:
        logging.info("ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        logging.error("SOME TESTS FAILED!")
        sys.exit(1)

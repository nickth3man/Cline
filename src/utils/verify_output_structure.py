"""
Verify and enforce the output structure for the Cline project.

This script scans the output directory and ensures each video folder follows
the strict project rule: each folder should only contain:
1. The original video file
2. The transcript file
3. The transcript summary

Usage:
    python verify_output_structure.py [output_directory]
"""

import os
import sys
import logging
import datetime
from pathlib import Path

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_filename = (
    log_dir
    / f"output_verification_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
    format="%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding="utf-8"),
    ],
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
            f"{base_name}_report.json",  # Processing report (allowed but optional)
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
        if all(f in allowed_files for f in remaining_files):
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


def verify_all_outputs(output_dir):
    """
    Verify and enforce the output structure for all video folders in the output directory.

    Args:
        output_dir: Path to the output directory containing video folders

    Returns:
        tuple: (success_count, failure_count)
    """
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        logging.error(f"Output directory does not exist: {output_dir}")
        return 0, 0

    success_count = 0
    failure_count = 0

    # Log environment information
    logging.debug(f"Current working directory: {os.getcwd()}")
    logging.debug(f"Python executable: {sys.executable}")
    logging.debug(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")

    # Log initial state
    logging.debug("Output directory contents:")
    for item in os.listdir(output_dir):
        logging.debug(f"  - {item}")

    # Get all subdirectories in the output directory
    video_folders = [
        d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ]

    if not video_folders:
        logging.warning(f"No video folders found in {output_dir}")
        return 0, 0

    logging.info(f"Found {len(video_folders)} video folders to verify")

    total_folders = 0
    compliant_folders = 0
    non_compliant_folders = 0

    for folder in video_folders:
        folder_path = os.path.join(output_dir, folder)

        # Log folder contents
        contents = os.listdir(folder_path)
        logging.debug(f"Contents of {folder}:")
        for item in contents:
            logging.debug(f"  - {item}")

        total_folders += 1
        logging.info(f"Verifying folder: {folder}")

        # Find the video file in the folder
        video_files = [
            f for f in contents if f.endswith((".mp4", ".mkv", ".webm", ".avi"))
        ]

        if not video_files:
            logging.warning(f"No video file found in {folder_path}, skipping")
            failure_count += 1
            non_compliant_folders += 1
            continue

        # Use the first video file found
        video_filename = video_files[0]

        logging.debug(f"Found video file: {video_filename}")

        # Check for transcript and summary
        base_name = os.path.splitext(video_filename)[0]
        transcript_file = f"{base_name}_corrected.md"
        summary_file = f"{base_name}_summary.md"

        required_files = {video_filename, transcript_file, summary_file}
        actual_files = set(contents)

        # Log detailed file analysis
        logging.debug(f"Required files: {required_files}")
        logging.debug(f"Actual files: {actual_files}")

        missing_files = required_files - actual_files
        extra_files = actual_files - required_files

        if missing_files:
            logging.error(f"Missing required files in {folder}: {missing_files}")
            failure_count += 1
            non_compliant_folders += 1
            continue

        if extra_files:
            logging.error(f"Extra files found in {folder}: {extra_files}")
            failure_count += 1
            non_compliant_folders += 1
            continue

        # Enforce output structure
        result = enforce_output_structure(folder_path, video_filename)

        if result:
            success_count += 1
            compliant_folders += 1
            logging.info(f"Folder {folder} is compliant with structure requirements")
        else:
            failure_count += 1
            non_compliant_folders += 1
            logging.error(f"✗ Failed to enforce structure for {folder}")

    # Log final statistics
    logging.info("\nVerification Summary:")
    logging.info(f"Total folders checked: {total_folders}")
    logging.info(f"Compliant folders: {compliant_folders}")
    logging.info(f"Non-compliant folders: {non_compliant_folders}")

    # Calculate compliance percentage
    if total_folders > 0:
        compliance_rate = (compliant_folders / total_folders) * 100
        logging.info(f"Compliance rate: {compliance_rate:.1f}%")

    return success_count, failure_count


if __name__ == "__main__":
    # Get output directory from command line or use default
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"

    logging.info(f"Starting output structure verification for: {output_dir}")

    success, failure = verify_all_outputs(output_dir)

    logging.info(f"Verification complete: {success} successful, {failure} failed")

    if failure > 0:
        logging.warning("Some folders could not be verified or enforced")
        sys.exit(1)
    else:
        logging.info("All folders successfully verified and enforced")
        sys.exit(0)

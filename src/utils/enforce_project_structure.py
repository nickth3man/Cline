"""
Enforce Project Structure

This script enforces the strict project structure requirements:
1. Consolidates output directories (merges yt_pipeline_output into output/)
2. Ensures each video folder only contains the required files:
   - Original video file
   - Transcript file (_corrected.md)
   - Summary file (_summary.md)
3. Removes all other files and subdirectories

Usage:
    python enforce_project_structure.py
"""

import os
import sys
import logging
import datetime
import shutil
from pathlib import Path

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_filename = (
    log_dir
    / f"structure_enforcement_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,
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
        video_dir = Path(video_dir)
        if not video_dir.exists() or not video_dir.is_dir():
            logging.error(f"Video directory does not exist: {video_dir}")
            return False

        base_name = Path(video_filename).stem

        # Define allowed files
        allowed_files = [
            video_filename,  # Original video
            f"{base_name}_corrected.md",  # Transcript
            f"{base_name}_summary.md",  # Summary
            f"{base_name}_report.json",  # Processing report (allowed but optional)
        ]

        removed_items = []

        # Log initial directory contents
        logging.debug(f"Initial contents of {video_dir}: {os.listdir(video_dir)}")

        # First, remove all subdirectories
        for item in video_dir.iterdir():
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                    removed_items.append(("directory", item.name))
                    logging.debug(f"Removed directory: {item}")
                except Exception as e:
                    logging.error(f"Failed to remove directory: {item}: {e}")
                    return False

        # Then remove all files except allowed ones
        for item in video_dir.iterdir():
            if item.is_file() and item.name not in allowed_files:
                try:
                    item.unlink()
                    removed_items.append(("file", item.name))
                    logging.debug(f"Removed file: {item}")
                except Exception as e:
                    logging.error(f"Failed to remove file: {item}: {e}")
                    return False

        # Log final directory contents
        logging.debug(f"Final contents of {video_dir}: {os.listdir(video_dir)}")

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
        remaining_files = [item.name for item in video_dir.iterdir()]
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


def consolidate_output_directories():
    """
    Consolidate output directories according to the project restructuring plan.
    Merges yt_pipeline_output into output/ directory.

    Returns:
        tuple: (success_count, failure_count)
    """
    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Log initial directory states
    logging.debug(
        f"Initial contents of yt_pipeline_output: {os.listdir('yt_pipeline_output') if os.path.exists('yt_pipeline_output') else 'Directory does not exist'}"
    )
    logging.debug(
        f"Initial contents of output: {os.listdir('output') if os.path.exists('output') else 'Directory does not exist'}"
    )

    # Check if yt_pipeline_output exists
    yt_output_dir = Path("yt_pipeline_output")
    if not yt_output_dir.exists() or not yt_output_dir.is_dir():
        logging.info("No yt_pipeline_output directory found, nothing to consolidate")
        return 0, 0

    success_count = 0
    failure_count = 0

    # Get all video folders in yt_pipeline_output
    video_folders = [d for d in yt_output_dir.iterdir() if d.is_dir()]

    if not video_folders:
        logging.info("No video folders found in yt_pipeline_output")
        return 0, 0

    logging.info(f"Found {len(video_folders)} video folders to consolidate")

    for folder in video_folders:
        target_folder = output_dir / folder.name

        # If target folder already exists, merge contents
        if target_folder.exists():
            logging.warning(f"Target folder already exists: {target_folder}")

            # Copy all files from source to target
            try:
                for item in folder.iterdir():
                    if item.is_file():
                        shutil.copy2(item, target_folder)
                        logging.debug(f"Copied file: {item} -> {target_folder}")

                success_count += 1
                logging.info(f"Successfully merged folder: {folder.name}")
            except Exception as e:
                failure_count += 1
                logging.error(f"Failed to merge folder {folder.name}: {e}")
        else:
            # Move the entire folder
            try:
                shutil.move(str(folder), str(target_folder))
                success_count += 1
                logging.debug(f"Moved folder: {folder} -> {target_folder}")
            except Exception as e:
                failure_count += 1
                logging.error(f"Failed to move folder {folder.name}: {e}")

    # Log final directory states
    logging.debug(
        f"Final contents of yt_pipeline_output: {os.listdir('yt_pipeline_output') if os.path.exists('yt_pipeline_output') else 'Directory does not exist'}"
    )
    logging.debug(f"Final contents of output: {os.listdir('output')}")

    # If yt_pipeline_output is now empty, remove it
    remaining_items = list(yt_output_dir.iterdir())
    if not remaining_items:
        try:
            yt_output_dir.rmdir()
            logging.info("Removed empty yt_pipeline_output directory")
        except Exception as e:
            logging.warning(f"Failed to remove empty yt_pipeline_output directory: {e}")

    return success_count, failure_count


def enforce_all_video_folders():
    """
    Enforce the output structure for all video folders in the output directory.

    Returns:
        tuple: (success_count, failure_count)
    """
    output_dir = Path("output")
    if not output_dir.exists() or not output_dir.is_dir():
        logging.error(f"Output directory does not exist: {output_dir}")
        return 0, 0

    success_count = 0
    failure_count = 0

    # Get all video folders in output directory
    video_folders = [d for d in output_dir.iterdir() if d.is_dir()]

    if not video_folders:
        logging.warning(f"No video folders found in {output_dir}")
        return 0, 0

    logging.info(f"Found {len(video_folders)} video folders to enforce structure")

    for folder in video_folders:
        # Find the video file in the folder
        video_files = [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in (".mp4", ".mkv", ".webm", ".avi")
        ]

        if not video_files:
            logging.warning(f"No video file found in {folder}, skipping")
            failure_count += 1
            continue

        # Use the first video file found
        video_filename = video_files[0].name

        logging.info(f"Enforcing structure for folder: {folder}")
        result = enforce_output_structure(folder, video_filename)

        if result:
            success_count += 1
            logging.info(f"✓ Successfully enforced structure for {folder.name}")
        else:
            failure_count += 1
            logging.error(f"✗ Failed to enforce structure for {folder.name}")

    return success_count, failure_count


def main():
    """Main function to enforce project structure."""
    logging.info("Starting project structure enforcement")

    # Log environment information
    logging.debug(f"Current working directory: {os.getcwd()}")
    logging.debug(f"Python executable: {sys.executable}")
    logging.debug(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")

    # Step 1: Consolidate output directories
    logging.info("Step 1: Consolidating output directories")
    success, failure = consolidate_output_directories()
    logging.info(f"Consolidation complete: {success} successful, {failure} failed")

    # Step 2: Enforce structure for all video folders
    logging.info("Step 2: Enforcing structure for all video folders")
    success, failure = enforce_all_video_folders()
    logging.info(
        f"Structure enforcement complete: {success} successful, {failure} failed"
    )

    # Print summary
    print(f"\nProject structure enforcement complete!")
    print(f"Log file: {log_filename}")

    if failure > 0:
        logging.warning("Some operations failed, check the log for details")
        return 1
    else:
        logging.info("All operations completed successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())

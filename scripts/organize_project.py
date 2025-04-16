#!/usr/bin/env python
"""
Project Reorganization Script

This script reorganizes the project file structure by:
1. Consolidating configuration files
2. Moving Python scripts to src/ directory
3. Consolidating output directories
4. Creating and populating a cache directory
5. Cleaning up temporary files
6. Removing redundant test directories
"""

import os
import shutil
import sys
import logging
from pathlib import Path
import datetime

# Setup logging
log_filename = f"reorganization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", log_filename)),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ["cache", "src/pipeline", "src/utils"]

    for directory in directories:
        dir_path = os.path.join(os.getcwd(), directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {directory}")

    logger.info("Directory setup complete")


def consolidate_config_files():
    """Consolidate configuration files to the config directory."""
    # Move root .env to config if it's different
    root_env = os.path.join(os.getcwd(), ".env")
    config_env = os.path.join(os.getcwd(), "config", ".env")

    if os.path.exists(root_env):
        if not os.path.exists(config_env):
            shutil.copy2(root_env, config_env)
            logger.info("Copied .env to config directory")
        else:
            # Check if files are different
            with open(root_env, "r") as root_file, open(config_env, "r") as config_file:
                root_content = root_file.read()
                config_content = config_file.read()

                if root_content != config_content:
                    # Create backup of config/.env
                    backup_path = os.path.join(
                        os.getcwd(),
                        "config",
                        f".env.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    )
                    shutil.copy2(config_env, backup_path)
                    logger.info(f"Created backup of config/.env at {backup_path}")

                    # Use the root .env as the canonical one
                    shutil.copy2(root_env, config_env)
                    logger.info("Updated config/.env with root .env content")

        os.remove(root_env)
        logger.info("Removed .env from root directory")

    # Handle requirements.txt
    root_req = os.path.join(os.getcwd(), "requirements.txt")
    config_req = os.path.join(os.getcwd(), "config", "requirements.txt")

    if os.path.exists(root_req) and os.path.exists(config_req):
        # Merge requirements files
        merged_requirements = set()

        with open(root_req, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    merged_requirements.add(line)

        with open(config_req, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    merged_requirements.add(line)

        # Write merged requirements to root
        with open(root_req, "w") as file:
            for req in sorted(merged_requirements):
                file.write(f"{req}\n")

        # Remove config requirements
        os.remove(config_req)
        logger.info(
            "Merged requirements.txt files and removed duplicate from config directory"
        )


def move_scripts_to_src():
    """Move Python scripts from root to appropriate src subdirectories."""
    # Define script mappings {script_name: destination_subdir}
    script_mappings = {
        "download_and_process.py": "pipeline",
        "process_downloaded_videos.py": "pipeline",
        "run_transcription_pipeline.py": "pipeline",
        "transcription_workflow.py": "transcription",
        "verify_output_structure.py": "utils",
        "enforce_project_structure.py": "utils",
        "test_cleanup_direct.py": "../tests",
        "test_cleanup_standalone.py": "../tests",
    }

    # Process each script
    for script, subdir in script_mappings.items():
        src_path = os.path.join(os.getcwd(), script)
        if os.path.exists(src_path):
            # Create destination directory if needed
            dest_dir = os.path.join(os.getcwd(), "src", subdir)
            if not os.path.isdir(dest_dir):
                os.makedirs(dest_dir)

            # Check for naming conflicts
            base_name = os.path.basename(script)
            dest_path = os.path.join(dest_dir, base_name)

            if os.path.exists(dest_path):
                # If file exists at destination, create a backup before overwriting
                backup_name = f"{base_name}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = os.path.join(dest_dir, backup_name)
                shutil.move(dest_path, backup_path)
                logger.info(f"Created backup of {dest_path} at {backup_path}")

            # Move the file
            shutil.copy2(src_path, dest_path)
            os.remove(src_path)
            logger.info(f"Moved {script} to {dest_dir}")


def consolidate_output_directories():
    """Consolidate output directories by moving yt_pipeline_output into output."""
    yt_pipeline_dir = os.path.join(os.getcwd(), "yt_pipeline_output")
    output_dir = os.path.join(os.getcwd(), "output")

    if os.path.exists(yt_pipeline_dir):
        # Move database file to output directory
        db_file = os.path.join(yt_pipeline_dir, "pipeline_output.db")
        if os.path.exists(db_file):
            dest_db_file = os.path.join(output_dir, "pipeline_output.db")

            # Handle existing file
            if os.path.exists(dest_db_file):
                backup_name = f"pipeline_output.db.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                backup_path = os.path.join(output_dir, backup_name)
                shutil.move(dest_db_file, backup_path)
                logger.info(f"Created backup of {dest_db_file} at {backup_path}")

            shutil.copy2(db_file, dest_db_file)
            logger.info(f"Moved database from {db_file} to {dest_db_file}")

        # Check if there's a root database file too
        root_db_file = os.path.join(os.getcwd(), "pipeline_output.db")
        if os.path.exists(root_db_file):
            # If the files are different, keep the newer one
            dest_db_file = os.path.join(output_dir, "pipeline_output.db")

            if os.path.exists(dest_db_file):
                root_mtime = os.path.getmtime(root_db_file)
                dest_mtime = os.path.getmtime(dest_db_file)

                if root_mtime > dest_mtime:
                    # Root is newer, create backup of output and replace
                    backup_name = f"pipeline_output.db.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    backup_path = os.path.join(output_dir, backup_name)
                    shutil.move(dest_db_file, backup_path)
                    shutil.copy2(root_db_file, dest_db_file)
                    logger.info(
                        f"Replaced output DB with newer root DB, backup at {backup_path}"
                    )
            else:
                # No file in output, just copy
                shutil.copy2(root_db_file, dest_db_file)
                logger.info(f"Copied root DB to output directory")

            # Remove root DB file
            os.remove(root_db_file)
            logger.info("Removed pipeline_output.db from root directory")

        # Remove yt_pipeline_output directory after consolidation
        shutil.rmtree(yt_pipeline_dir)
        logger.info("Removed yt_pipeline_output directory after consolidation")


def setup_cache():
    """Set up cache directory and move cache files there."""
    cache_dir = os.path.join(os.getcwd(), "cache")

    # Move .openrouter_models_cache.json to cache
    router_cache = os.path.join(os.getcwd(), ".openrouter_models_cache.json")
    if os.path.exists(router_cache):
        dest_path = os.path.join(cache_dir, "openrouter_models_cache.json")
        shutil.copy2(router_cache, dest_path)
        os.remove(router_cache)
        logger.info("Moved .openrouter_models_cache.json to cache directory")

    # Create a .gitignore file for the cache directory
    gitignore_path = os.path.join(cache_dir, ".gitignore")
    with open(gitignore_path, "w") as file:
        file.write("# Ignore all files in this directory\n*\n!.gitignore\n")
    logger.info("Created .gitignore in cache directory")


def clean_temp_downloads():
    """Clean up temporary downloads directory."""
    temp_dir = os.path.join(os.getcwd(), "temp_downloads")

    if os.path.exists(temp_dir):
        # Remove partial downloads
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if file.endswith(".part") or file.endswith(".ytdl"):
                os.remove(file_path)
                logger.info(f"Removed partial download: {file}")

        # Create .gitignore for temp_downloads
        gitignore_path = os.path.join(temp_dir, ".gitignore")
        with open(gitignore_path, "w") as file:
            file.write("# Ignore all files in this directory\n*\n!.gitignore\n")
        logger.info("Created .gitignore in temp_downloads directory")


def consolidate_tests():
    """Consolidate test directories and ensure consistent naming."""
    test_dir = os.path.join(os.getcwd(), "test")
    tests_dir = os.path.join(os.getcwd(), "tests")

    # If test directory exists and is empty, remove it
    if os.path.exists(test_dir) and not os.listdir(test_dir):
        os.rmdir(test_dir)
        logger.info("Removed empty test directory")

    # Ensure all test files in tests directory follow test_*.py naming
    if os.path.exists(tests_dir):
        for file in os.listdir(tests_dir):
            if (
                file.endswith(".py")
                and not file.startswith("test_")
                and not file == "conftest.py"
            ):
                old_path = os.path.join(tests_dir, file)
                new_name = f"test_{file}"
                new_path = os.path.join(tests_dir, new_name)

                # Check if destination already exists
                if os.path.exists(new_path):
                    backup_name = f"{new_name}.backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    backup_path = os.path.join(tests_dir, backup_name)
                    shutil.move(new_path, backup_path)
                    logger.info(f"Created backup of {new_path} at {backup_path}")

                shutil.move(old_path, new_path)
                logger.info(f"Renamed {file} to {new_name} for consistent test naming")


def enforce_output_structure():
    """Enforce the strict output structure requirements."""
    output_dir = os.path.join(os.getcwd(), "output")

    if not os.path.exists(output_dir):
        logger.warning(
            "Output directory doesn't exist. Skipping structure enforcement."
        )
        return

    # Iterate through video folders
    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)

        # Skip if not a directory or if it's not a video folder
        if not os.path.isdir(folder_path):
            continue

        logger.info(f"Enforcing structure for {folder_name}")

        # Get list of required file patterns
        video_file_pattern = f"{folder_name}.mp4"
        transcript_pattern = f"{folder_name}_corrected.md"
        summary_pattern = f"{folder_name}_summary.md"

        # Check all files in the directory
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)

            # Check if file is allowed (one of the three required files)
            is_allowed = (
                item == video_file_pattern
                or item == transcript_pattern
                or item == summary_pattern
            )

            # If not allowed and is a directory or file, remove it
            if not is_allowed:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    logger.info(f"Removed subdirectory {item} from {folder_name}")
                else:
                    os.remove(item_path)
                    logger.info(f"Removed disallowed file {item} from {folder_name}")


def main():
    """Main execution function."""
    try:
        logger.info("Starting project reorganization")

        # Step 1: Setup directories
        setup_directories()

        # Step 2: Consolidate config files
        consolidate_config_files()

        # Step 3: Move scripts to src
        move_scripts_to_src()

        # Step 4: Consolidate output directories
        consolidate_output_directories()

        # Step 5: Setup cache
        setup_cache()

        # Step 6: Clean temp downloads
        clean_temp_downloads()

        # Step 7: Consolidate tests
        consolidate_tests()

        # Step 8: Enforce output structure
        enforce_output_structure()

        logger.info("Project reorganization completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error during reorganization: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

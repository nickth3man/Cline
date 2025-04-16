#!/usr/bin/env python
"""
Tool to analyze and fix circular imports in the Cline project.
"""

import os
import sys
import importlib
import re
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def scan_file_for_imports(file_path):
    """Scan a file for import statements and return them."""
    import_pattern = re.compile(r"^(?:from\s+(\S+)\s+import|import\s+([^,\s]+))")
    circular_pattern = re.compile(r"from\s+(\S+)\s+import")

    imports = []
    potential_circular_imports = []

    # Try different encodings
    encodings = ["utf-8", "latin1", "cp1252"]

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Check for import statements
                    match = import_pattern.match(line)
                    if match:
                        module = match.group(1) or match.group(2)
                        imports.append((module, line_num, line))

                        # Check for potential circular imports (module importing from same directory)
                        circular_match = circular_pattern.match(line)
                        if circular_match:
                            imported_module = circular_match.group(1)
                            module_name = os.path.basename(file_path).replace(".py", "")
                            module_dir = os.path.dirname(file_path)

                            # Check if the imported module matches this file's name
                            if imported_module.endswith(
                                module_name
                            ) or module_name in imported_module.split("."):
                                potential_circular_imports.append(
                                    (imported_module, line_num, line)
                                )
            # If we successfully read the file, break out of the loop
            break
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                # If this is the last encoding we're trying, log the error
                logging.warning(
                    f"Could not decode file {file_path} with any of the attempted encodings"
                )
                return [], []
            # Otherwise try the next encoding
            continue
        except Exception as e:
            logging.warning(f"Error reading file {file_path}: {e}")
            return [], []

    return imports, potential_circular_imports


def analyze_project_imports(project_root):
    """Analyze imports across the entire project and identify potential circular dependencies."""
    python_files = []

    # Find all Python files in the project
    for root, _, files in os.walk(project_root):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    import_map = {}
    potential_circular_imports = []

    # Analyze imports in each file
    for file_path in python_files:
        rel_path = os.path.relpath(file_path, project_root)
        imports, circular_imports = scan_file_for_imports(file_path)
        import_map[rel_path] = imports

        if circular_imports:
            potential_circular_imports.append((rel_path, circular_imports))

    return import_map, potential_circular_imports


def fix_transcription_workflow_circular_import(project_root):
    """Fix the circular import in transcription_workflow.py."""
    transcription_workflow_path = os.path.join(
        project_root, "src", "transcription", "transcription_workflow.py"
    )

    if not os.path.exists(transcription_workflow_path):
        logging.error(f"File not found: {transcription_workflow_path}")
        return False

    # Try different encodings
    encodings = ["utf-8", "latin1", "cp1252"]
    content = None

    for encoding in encodings:
        try:
            # Read the file content
            with open(transcription_workflow_path, "r", encoding=encoding) as f:
                content = f.read()
            # If we successfully read the file, break out of the loop
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.error(f"Error reading file {transcription_workflow_path}: {e}")
            return False

    if content is None:
        logging.error(
            f"Could not decode file {transcription_workflow_path} with any of the attempted encodings"
        )
        return False

    # Check for self-import
    self_import_pattern = r"from\s+transcription\.transcription_workflow\s+import"
    if not re.search(self_import_pattern, content):
        logging.info(f"No circular import found in {transcription_workflow_path}")
        return False

    # Create backup of the original file
    backup_path = transcription_workflow_path + ".bak"
    shutil.copy2(transcription_workflow_path, backup_path)
    logging.info(f"Created backup at {backup_path}")

    # Remove the self-import
    modified_content = re.sub(
        self_import_pattern + r"\s*\(\s*[^)]*\)", "# REMOVED CIRCULAR IMPORT", content
    )

    # Write the modified content back to the file
    with open(transcription_workflow_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    logging.info(f"Fixed circular import in {transcription_workflow_path}")
    return True


def find_runner_script(project_root):
    """Find a script that can be used to test importing the transcription workflow module."""
    for root, _, files in os.walk(project_root):
        for file in files:
            if (
                file.endswith(".py")
                and "test" in file.lower()
                and "transcription" in file.lower()
            ):
                return os.path.join(root, file)
    return None


def main():
    if len(sys.argv) < 2:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        project_root = os.path.abspath(sys.argv[1])

    logging.info(f"Analyzing imports in project: {project_root}")

    # Analyze project imports
    _, circular_imports = analyze_project_imports(project_root)

    if not circular_imports:
        logging.info("No potential circular imports found.")
        return 0

    logging.info(
        f"Found {len(circular_imports)} files with potential circular imports:"
    )
    for file_path, imports in circular_imports:
        logging.info(f"File: {file_path}")
        for module, line_num, line in imports:
            logging.info(f"  Line {line_num}: {line}")

    # Fix the transcription_workflow circular import
    if any("transcription_workflow.py" in file for file, _ in circular_imports):
        fixed = fix_transcription_workflow_circular_import(project_root)
        if fixed:
            logging.info("Fixed circular import in transcription_workflow.py")

            # Find a test script
            test_script = find_runner_script(project_root)
            if test_script:
                logging.info(
                    f"You can test the fix by running: python {os.path.relpath(test_script, project_root)}"
                )
            else:
                logging.info(
                    "You can test the fix by running your application's main script"
                )
        else:
            logging.warning("Could not fix circular import automatically")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Script to diagnose and fix Python environment issues.
This script helps identify which Python interpreter is being used and provides
instructions to ensure the correct project environment is activated.
"""

import os
import sys
import subprocess
import shutil


def print_separator():
    print("-" * 80)


def main():
    print_separator()
    print("Python Environment Diagnostic Tool")
    print_separator()

    # Display current Python executable
    print(f"Current Python executable: {sys.executable}")

    # Display sys.path
    print("\nPython path (sys.path):")
    for path in sys.path:
        print(f"  - {path}")

    # Display environment variables
    print("\nRelevant environment variables:")
    print(f"  PATH: {os.environ.get('PATH', 'Not set')}")
    print(f"  VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")

    # Determine if we're in the correct environment
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_venv = os.path.join(project_dir, "venv")
    project_venv_scripts = os.path.join(project_venv, "Scripts")
    project_python = os.path.join(project_venv_scripts, "python.exe")

    print("\nProject environment check:")
    print(f"  Project directory: {project_dir}")
    print(f"  Project venv: {project_venv}")
    print(f"  Project Python: {project_python}")

    using_correct_env = sys.executable.lower() == project_python.lower()
    print(f"  Using correct Python interpreter: {'Yes' if using_correct_env else 'No'}")

    # Check if yt-dlp is available
    yt_dlp_path = os.path.join(project_venv_scripts, "yt-dlp.exe")
    has_yt_dlp = os.path.exists(yt_dlp_path)
    print(f"  yt-dlp available in project venv: {'Yes' if has_yt_dlp else 'No'}")

    # Provide fix instructions
    print_separator()
    print("DIAGNOSIS AND RECOMMENDATIONS:")

    if using_correct_env:
        print("✓ You are using the correct Python interpreter.")
    else:
        print("✗ You are NOT using the correct Python interpreter.")
        print("\nTo fix this issue:")

        # Check if VS Code is being used (common scenario)
        if os.path.exists(os.path.join(project_dir, ".vscode")):
            print("\n1. VS Code specific fixes:")
            print("   a. Open command palette (Ctrl+Shift+P)")
            print("   b. Type 'Python: Select Interpreter'")
            print("   c. Select the interpreter from your project's venv:")
            print(f"      {project_python}")
            print("   d. Create or update .vscode/settings.json with:")
            print(
                '      {\n        "python.defaultInterpreterPath": '
                + f'"{project_python.replace(os.sep, "/")}"\n      }}'
            )

        print("\n2. Terminal specific fixes:")
        print(
            "   a. Always activate the correct virtual environment before running scripts:"
        )
        print(f"      * Windows PowerShell: {project_venv_scripts}\\Activate.ps1")
        print(f"      * Windows CMD: {project_venv_scripts}\\activate.bat")

        print("\n3. Script wrapper fix:")
        print(
            "   Create wrapper scripts that explicitly use the correct Python interpreter:"
        )
        print(f"      * Example: {project_python} your_script.py [arguments]")

        # Option to create wrapper scripts
        print("\n4. Create wrapper batch files:")
        print("   The following wrapper scripts will be created in the project root:")
        print("   * run_pipeline.bat - Runs the pipeline with the correct interpreter")
        print(
            "   * verify_structure.bat - Verifies output structure with correct interpreter"
        )

        if (
            input("\nWould you like to create these wrapper scripts? (y/n): ").lower()
            == "y"
        ):
            # Create run_pipeline.bat
            with open(os.path.join(project_dir, "run_pipeline.bat"), "w") as f:
                f.write(f"@echo off\n")
                f.write(f"echo Running pipeline with correct Python interpreter...\n")
                f.write(
                    f'"{project_python}" "%~dp0src\\pipeline\\run_full_pipeline.py" %*\n'
                )

            # Create verify_structure.bat
            with open(os.path.join(project_dir, "verify_structure.bat"), "w") as f:
                f.write(f"@echo off\n")
                f.write(
                    f"echo Verifying output structure with correct Python interpreter...\n"
                )
                f.write(
                    f'"{project_python}" "%~dp0src\\utilities\\verify_output_structure.py" %*\n'
                )

            print("Wrapper scripts created successfully!")

    print_separator()
    print("For immediate testing:")
    print(
        f"Run: {project_python} -c \"import sys, yt_dlp; print('Python:', sys.executable); print('yt-dlp version:', yt_dlp.version.__version__)\""
    )
    print_separator()


if __name__ == "__main__":
    main()

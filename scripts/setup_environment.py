import os
import subprocess
import sys

VENV_DIR = "venv"
SPACY_MODEL = "en_core_web_sm"


def run_command(command, check=True):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        if check:
            print(
                f"Command {' '.join(command)} failed with exit code {result.returncode}"
            )
            sys.exit(result.returncode)
    return result


def create_virtualenv():
    if not os.path.isdir(VENV_DIR):
        print(f"Creating virtual environment in {VENV_DIR}...")
        run_command([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print(f"Virtual environment {VENV_DIR} already exists.")


def activate_virtualenv():
    if os.name == "nt":
        activate_script = os.path.join(VENV_DIR, "Scripts", "activate")
    else:
        activate_script = os.path.join(VENV_DIR, "bin", "activate")
    print(f"To activate the virtual environment, run:\nsource {activate_script}")


def install_requirements():
    pip_executable = os.path.join(
        VENV_DIR, "Scripts" if os.name == "nt" else "bin", "pip"
    )
    print("Installing dependencies from requirements.txt...")
    run_command([pip_executable, "install", "-r", "requirements.txt"])


def download_spacy_model():
    try:
        import spacy

        spacy.load(SPACY_MODEL)
        print(f"SpaCy model '{SPACY_MODEL}' is already installed.")
    except ImportError:
        print(
            "SpaCy is not installed. Please run the setup script after installing dependencies."
        )
        sys.exit(1)
    except OSError:
        print(f"Downloading SpaCy model '{SPACY_MODEL}'...")
        run_command(
            [
                os.path.join(
                    VENV_DIR, "Scripts" if os.name == "nt" else "bin", "python"
                ),
                "-m",
                "spacy",
                "download",
                SPACY_MODEL,
            ]
        )


def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        print("FFmpeg is installed and accessible.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "FFmpeg not found or not accessible. Please install FFmpeg and add it to your system PATH."
        )


def check_hf_token():
    if os.getenv("HF_TOKEN"):
        print("Hugging Face token (HF_TOKEN) is set.")
    else:
        print(
            "Warning: Hugging Face token (HF_TOKEN) is not set. Required for pyannote.audio model downloads."
        )


def main():
    create_virtualenv()
    activate_virtualenv()
    install_requirements()
    download_spacy_model()
    check_ffmpeg()
    check_hf_token()
    print(
        "\nSetup complete. Please activate the virtual environment before running the pipeline."
    )


if __name__ == "__main__":
    main()

import importlib
import subprocess
import sys
import os

REQUIRED_PACKAGES = [
    "yt_dlp",
    "openai",
    "spacy",
    "pyannote.audio",
    "torch",
    "requests",
    "python_dotenv",
]

SPACY_MODEL = "en_core_web_sm"


def check_python_packages():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_spacy_model():
    import spacy

    try:
        spacy.load(SPACY_MODEL)
        return True
    except OSError:
        return False


def check_hf_token():
    return bool(os.getenv("HF_TOKEN"))


def main():
    print("Checking dependencies...\n")

    missing_pkgs = check_python_packages()
    if missing_pkgs:
        print("Missing Python packages:")
        for pkg in missing_pkgs:
            print(f"  - {pkg}")
        print("Please install them using pip, e.g.:")
        print(f"  pip install {' '.join(missing_pkgs)}\n")
    else:
        print("All required Python packages are installed.")

    if not check_ffmpeg():
        print(
            "FFmpeg not found or not working. Please install FFmpeg and ensure it is in your system PATH.\n"
        )
    else:
        print("FFmpeg is installed and accessible.")

    if not check_spacy_model():
        print(f"SpaCy model '{SPACY_MODEL}' not found. Please install it by running:")
        print(f"  python -m spacy download {SPACY_MODEL}\n")
    else:
        print(f"SpaCy model '{SPACY_MODEL}' is installed.")

    if not check_hf_token():
        print(
            "Hugging Face token (HF_TOKEN) not set in environment variables. Required for pyannote.audio model downloads.\n"
        )
    else:
        print("Hugging Face token (HF_TOKEN) is set.")

    if not missing_pkgs and check_ffmpeg() and check_spacy_model() and check_hf_token():
        print("\nAll dependencies are satisfied. You are ready to run the pipeline.")
        sys.exit(0)
    else:
        print("\nPlease address the above issues before running the pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()

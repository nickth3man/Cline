> Each video folder must ONLY contain: (1) the original video file, (2) the transcript file with _corrected.md suffix, (3) the summary file with _summary.md suffix. No other files or folders are allowed.

# Active Context

## Current Focus

- **OpenRouter-Aligned LLM Model Selection and Workflow:**
  - Transcription: Performed using the official OpenAI Whisper API (`whisper-1`) with `OPENAI_API_KEY` (not via OpenRouter, due to lack of endpoint).
  - Sentence splitting and cleanup: SpaCy (required).
  - LLM correction and summarization: User-selectable from all OpenRouter LLMs, with live price calculation and advanced model selection (search, variants, provider routing, error handling, model details).
  - Diarization: Performed locally using `pyannote.audio`.
  - Output: Per-video folders with audio, raw transcript, LLM-labeled transcript, summary, metadata (CSV/JSON), and HTML transcript reader.
  - UI: Correction and summarization dropdowns are populated live from OpenRouter, support search/filter, show model details (provider, context, price, variants, tool calling, structured outputs), and allow advanced selection (variants, provider routing).
  - Robust error handling: All API/network/cache errors are surfaced in the GUI, with clear user feedback and troubleshooting steps.

**Technical Note:**  
OpenRouter does not provide a Whisper (audio transcription) API endpoint. The pipeline uses the official OpenAI API for transcription, SpaCy for sentence splitting, `pyannote.audio` for local diarization, and OpenRouter for all LLM-based tasks (correction, summarization). Both `OPENAI_API_KEY` and `OPENROUTER_API_KEY` must be set in the environment. A Hugging Face token (`HF_TOKEN`) is required for `pyannote.audio` model downloads.

## Recent Changes

- **Resolved Python Environment Issue:**
  - Fixed the problem of incorrect Python interpreter (`C:\Users\nicki\Documents\Cline\MCP\browser-use\.venv\Scripts\python.exe`) being used instead of the project's venv.
  - Created diagnostic script (`fix_python_environment.py`) to identify environment issues and generate solutions.
  - Implemented wrapper batch files (`run_pipeline.bat` and `verify_structure.bat`) that explicitly use the correct Python interpreter.
  - Created a comprehensive Python wrapper script (`python_wrapper.py`) to handle module imports correctly.
  - Fixed circular imports in the codebase, particularly in `transcription_workflow.py`.
  - Implemented missing functions and corrected module import structure throughout the codebase.
  - The pipeline now successfully runs with the correct Python interpreter and can locate all required modules.

- Enhanced logging across key scripts (`download_and_process.py`, `workflow_logic.py`, `enforce_project_structure.py`, `verify_output_structure.py`) to include detailed environment information, system state, directory contents, and specific error contexts for improved debugging.
- Added DEBUG level logging with function names and line numbers for granular tracing.
- Implemented a robust, direct YouTube playlist handler capable of extracting video information through multiple fallback methods.
- Improved error handling in the pipeline to gracefully skip unavailable videos and continue processing.
- Enhanced `yt-dlp` integration to directly download and convert audio to WAV format in a single step.
- Added multiple fallback methods for audio extraction with different format selections and user agents.
- Added comprehensive try/except blocks around each processing stage to prevent pipeline failures.
- Implemented proper metadata extraction and JSON reporting for processing statistics.

## Next Steps and Considerations

- **Implement a comprehensive dependency check**: Add a startup script to verify all required Python and system dependencies, surfacing clear errors in the GUI and logs.
- **Enhance environment setup documentation**: Update usage guides and troubleshooting docs to help users avoid interpreter and dependency issues.
- **Develop an automated environment setup script**: Create a script to set up the virtual environment, install dependencies, and download required models.
- **Monitor for new OpenRouter features and API changes**: Establish a routine for checking updates and aligning the pipeline and documentation.
- **Expand test coverage**: Add tests for dependency errors, GUI integration, and edge cases.
- **Maintain memory bank alignment**: After each major change, update all relevant memory bank files to reflect the current implementation and learnings.

- **Investigate and fix playlist URL handling and video extraction logic**: Address the current pipeline failure to download videos due to invalid URLs.
- **Add robust error handling and fallback mechanisms**: Ensure the pipeline gracefully handles unavailable videos and extraction failures.
- **Consider updating yt-dlp to the latest version**: Improve extractor support and compatibility.
- **Add pre-run validation of playlist and video URLs**: Prevent pipeline runs with invalid or unavailable videos.

## Design and Technical Patterns

- Clean Python environment management with explicit interpreter selection and path handling.
- Modular code structure with clear separation of concerns.
- Comprehensive error handling and logging throughout the pipeline.
- Use of wrapper scripts to isolate environment-specific configurations.
- Multiple fallback methods for critical operations like YouTube download.
- User experience focused on clarity, transparency of cost, and ease of use.
- Strict alignment with OpenRouter's API, model selection, and advanced usage documentation.

## Project Insights

- Python virtual environment management is critical for consistent script execution.
- Explicit path handling is necessary when mixing absolute and relative imports.
- Circular imports can cause subtle and hard-to-diagnose issues in Python modules.
- Wrapper scripts provide a reliable way to ensure consistent environment configuration.

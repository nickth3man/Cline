# Complete All Tasks: Full Pipeline Execution & Validation

## Objective

Execute and validate the entire YouTube playlist processing pipeline, ensuring all requirements, enhancements, and documentation are fulfilled as described in the project memory bank.

---

## Steps

1. **Download Videos**
   - Download all videos from the playlist:  
     `https://www.youtube.com/playlist?list=PLTSh1Wa3gjplaOdhzJzzv6Tzh3KbiZsrX`
   - Use the fixed output directory: `output/`
   - Each video should have its own subfolder, named predictably.

2. **For Each Video:**
   - **Audio Extraction:**  
     - Use `yt-dlp` to extract audio and save in the video’s folder.
   - **Transcription:**  
     - Use the official OpenAI Whisper API (`whisper-1`) with `OPENAI_API_KEY` (not OpenRouter).
   - **Sentence Splitting:**  
     - Use SpaCy for sentence segmentation and cleanup (required).
   - **LLM Correction & Summarization:**  
     - Use OpenRouter for all LLM-based tasks (correction, summarization).
     - User can select any available OpenRouter LLM; UI must support live model fetching, search/filter, variants, provider routing, model details, and real-time price calculation.
   - **Pseudo-Diarization:**  
     - Use OpenRouter LLM to infer and assign speaker labels to the transcript after transcription, based on context and timestamps (no Hugging Face/pyannote required).
   - **Output Generation:**  
     - Save all outputs in the video’s folder:
       - Audio file
       - Raw transcript
       - LLM-labeled transcript
       - Summary
       - Metadata (CSV/JSON)
       - HTML transcript reader

3. **Validation & Logging**
   - Validate all outputs for completeness and correctness.
   - Log results and errors per video.
   - Surface all errors and troubleshooting steps in the GUI.

4. **Documentation & Memory Bank**
   - Update documentation and all relevant memory bank files with findings, learnings, and any changes to the workflow or requirements.

5. **Ongoing Monitoring & Enhancements**
   - Monitor for new OpenRouter features (e.g., new model types, advanced routing) and update the workflow as needed.
   - Maintain documentation and memory bank alignment with OpenRouter’s evolving API and best practices.
   - Consider further enhancements:
     - Exporting combined metadata.
     - More granular progress/status reporting.
     - Additional output formats or downstream integrations.

---

## Requirements

- **Environment Variables:**  
  - `OPENAI_API_KEY` (for Whisper transcription)
  - `OPENROUTER_API_KEY` (for LLM tasks)
- **Fixed Models/Tools:**  
  - Transcription: OpenAI Whisper (`whisper-1`)
  - Sentence splitting: SpaCy
  - LLM steps: User-selectable from OpenRouter (live-fetched)
  - Pseudo-diarization: OpenRouter LLM
- **Output Structure:**  
  - All assets grouped per video in `output/`
  - Consistent, predictable naming

---

## Notes

- No user options for transcription or sentence splitting models.
- All LLM steps are user-selectable from OpenRouter, with advanced selection and live pricing.
- Pseudo-diarization is always performed using OpenRouter LLM (no Hugging Face/pyannote).
- All error handling and user feedback must be surfaced in the GUI.
- Documentation and memory bank must be kept up to date after each run.

---

**This task, when completed, will fulfill all current project requirements and ensure the pipeline is robust, user-friendly, and fully aligned with OpenRouter’s best practices.**

import os
import json
import logging
import re
import yt_dlp
from src.transcription.transcription_workflow import (
    extract_or_convert_audio,
    transcribe_audio,
    diarize_speakers,
    correct_transcript,
)
from src.utils import workflow_logic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[^\w\-. ]', '', filename)  # Allow dots and hyphens
    sanitized = re.sub(r'\\s+', '_', sanitized)  # Replace whitespace sequences with underscore
    return sanitized[:150]  # Limit length

def generate_html_reader(corrected_transcript: str, output_path: str):
    html_content = f\"\"\"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Transcript Reader</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; }}
</style>
</head>
<body>
<h1>Transcript Reader</h1>
<pre>{corrected_transcript}</pre>
</body>
</html>\"\"\"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logging.info(f"HTML transcript reader saved: {output_path}")

def run_pipeline(playlist_url: str, output_base_dir: str, correction_models: list, summarization_model: str):
    # Fetch playlist info
    ydl_opts_info = {'quiet': True, 'extract_flat': 'in_playlist', 'skip_download': True}
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        if 'entries' not in info or not info['entries']:
            raise ValueError("No videos found in the playlist or playlist is invalid.")
        video_list = info['entries']
    logging.info(f"Found {len(video_list)} videos in playlist.")

    for index, video in enumerate(video_list):
        video_title = video.get('title', f'video_{index}')
        video_url = video.get('url') or video.get('webpage_url') or video.get('id')
        sanitized_title = sanitize_filename(video_title)
        video_folder = os.path.join(output_base_dir, f\"{index:03d}_{sanitized_title}\")
        os.makedirs(video_folder, exist_ok=True)

        logging.info(f\"Processing video {index+1}/{len(video_list)}: {video_title}\")

        # Download video
        download_path = os.path.join(video_folder, f\"{sanitized_title}.mp4\")
        ydl_opts_download = {
            'format': 'bestaudio/best',
            'outtmpl': download_path,
            'quiet': True,
            'noprogress': True,
            'http_headers': {'User-Agent': 'Mozilla/5.0'},
        }
        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl_download:
            ydl_download.download([video_url])
        logging.info(f\"Downloaded video to {download_path}\")

        # Extract/convert audio
        wav_audio_path = os.path.join(video_folder, f\"{sanitized_title}_audio.wav\")
        extract_or_convert_audio(download_path, wav_audio_path)

        # Transcribe audio
        raw_transcript = transcribe_audio(wav_audio_path)
        raw_transcript_path = os.path.join(video_folder, f\"{sanitized_title}_raw_transcript.txt\")
        with open(raw_transcript_path, \"w\", encoding=\"utf-8\") as f:
            f.write(raw_transcript)
        logging.info(f\"Raw transcript saved: {raw_transcript_path}\")

        # Diarize speakers
        diarization_result = diarize_speakers(wav_audio_path)
        diarization_path = os.path.join(video_folder, f\"{sanitized_title}_diarization.json\")
        with open(diarization_path, \"w\", encoding=\"utf-8\") as f:
            json.dump(diarization_result, f, indent=2)
        logging.info(f\"Diarization results saved: {diarization_path}\")

        # Correct transcript with multiple correction models
        corrected_transcripts = {}
        for model_id in correction_models:
            try:
                corrected = correct_transcript(raw_transcript, diarization_result, correction_model=model_id)
                corrected_transcripts[model_id] = corrected
                corrected_path = os.path.join(video_folder, f\"{sanitized_title}_corrected_{model_id.replace('/', '_')}.md\")
                with open(corrected_path, \"w\", encoding=\"utf-8\") as f:
                    f.write(corrected)
                logging.info(f\"Corrected transcript saved for model {model_id}: {corrected_path}\")
            except Exception as e:
                logging.error(f\"Correction failed for model {model_id} on video {video_title}: {e}\")

        # Summarize transcript using summarization model
        # Use the first corrected transcript if available, else raw transcript
        transcript_to_summarize = next(iter(corrected_transcripts.values()), raw_transcript)
        try:
            summary = workflow_logic.summarize_transcript(transcript_to_summarize, summarization_model)
            summary_path = os.path.join(video_folder, f\"{sanitized_title}_summary_{summarization_model.replace('/', '_')}.txt\")
            with open(summary_path, \"w\", encoding=\"utf-8\") as f:
                f.write(summary)
            logging.info(f\"Summary saved: {summary_path}\")
        except Exception as e:
            logging.error(f\"Summarization failed for video {video_title}: {e}\")

        # Generate HTML transcript reader for the first corrected transcript
        if corrected_transcripts:
            first_model_id = list(corrected_transcripts.keys())[0]
            html_path = os.path.join(video_folder, f\"{sanitized_title}_transcript_reader.html\")
            generate_html_reader(corrected_transcripts[first_model_id], html_path)

    logging.info(\"Pipeline processing complete for all videos.\")

if __name__ == \"__main__\":
    import sys
    if len(sys.argv) < 2:
        print(\"Usage: python run_full_pipeline.py <playlist_url>\")
        sys.exit(1)
    playlist_url = sys.argv[1]
    output_dir = \"yt_pipeline_output\"
    correction_models = [\"openai/gpt-4.1-mini\", \"anthropic/claude-3-haiku-20240307\"]
    summarization_model = \"openai/gpt-4.1-mini\"
    run_pipeline(playlist_url, output_dir, correction_models, summarization_model)

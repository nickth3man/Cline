import os
import argparse
from yt_dlp import YoutubeDL
import openai

def sanitize_filename(name: str) -> str:
    # Remove or replace characters that are invalid in folder names
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, '_')
    return name.strip()

def transcribe_audio_with_openai_whisper(audio_file_path: str, transcript_save_path: str):
    """
    Transcribe audio using the official OpenAI Whisper API (whisper-1).
    Saves the raw transcript text to transcript_save_path.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    with open(transcript_save_path, "w", encoding="utf-8") as f:
        f.write(transcript)

def download_playlist(playlist_url: str, video_ids=None, output_path="output"):
    ydl_opts = {
        'ignoreerrors': True,
        'quiet': False,
        'no_warnings': True,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegMerger',
        }],
        'restrictfilenames': True,  # Restrict filenames to ASCII and avoid special chars
        'noplaylist': False,
        'playlistend': None,
        'continuedl': True,
        'nooverwrites': False,
        'writesubtitles': False,
        'writeautomaticsub': False,
        'writethumbnail': False,
        'writeinfojson': False,
        'writedescription': False,
        'writeannotations': False,
        'writedmca': False,
        'writecomments': False,
        'outtmpl': os.path.join(output_path, '%(title)s/%(title)s.%(ext)s'),
        'postprocessor_args': ['-movflags', '+faststart'],
    }

    with YoutubeDL(ydl_opts) as ydl:
        # Extract playlist info to get video titles for folder sanitization
        info_dict = ydl.extract_info(playlist_url, download=False)
        entries = info_dict.get('entries', [])
        
        # Filter entries if video_ids is provided
        if video_ids:
            entries = [entry for entry in entries if entry and entry.get('id') in video_ids]
            
        for entry in entries:
            if entry is None:
                continue
            title = entry.get('title', 'unknown_title')
            video_id = entry.get('id', 'unknown_id') # Extract video ID
            sanitized_title = sanitize_filename(title)
            # Override output template for this video to use sanitized folder name
            ydl.params['outtmpl'] = os.path.join(output_path, f'{sanitized_title}/{sanitized_title}.%(ext)s')
            print(f'VIDEO_STATUS:{video_id}:Downloading video: {title}') # Structured status
            try:
                ydl.download([entry['webpage_url']])
                print(f'VIDEO_STATUS:{video_id}:Video downloaded') # Structured status
                # After download, transcribe audio
                video_folder = os.path.join(output_path, sanitized_title)
                # Find the downloaded video file path (mp4)
                video_file_path = os.path.join(video_folder, f"{sanitized_title}.mp4")
                if not os.path.exists(video_file_path):
                    print(f"VIDEO_ERROR:{video_id}:Video file not found for transcription: {video_file_path}") # Structured error
                    continue
                transcript_path = os.path.join(video_folder, f"{sanitized_title}_transcript.txt")
                print(f"VIDEO_STATUS:{video_id}:Transcribing audio for video: {title}") # Structured status
                transcribe_audio_with_openai_whisper(video_file_path, transcript_path)
                print(f"VIDEO_STATUS:{video_id}:Transcript saved to: {transcript_path}") # Structured status
                # Delete video file after successful transcription
                if os.path.exists(video_file_path):
                    os.remove(video_file_path)
                    print(f"VIDEO_STATUS:{video_id}:Deleted video file: {video_file_path}") # Structured status
            except Exception as e:
                print(f'VIDEO_ERROR:{video_id}:Error processing {title}: {e}') # Structured error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download videos from a YouTube playlist")
    parser.add_argument("--playlist_url", required=True, help="URL of the YouTube playlist")
    parser.add_argument("--video_ids", help="Comma-separated list of video IDs to download")
    parser.add_argument("--output_path", default="output", help="Path to save downloaded videos")
    
    args = parser.parse_args()
    
    # Process video_ids if provided
    video_id_list = None
    if args.video_ids:
        video_id_list = args.video_ids.split(",")
        
    download_playlist(args.playlist_url, video_id_list, args.output_path)

import sys
import yt_dlp
import json

playlist_url = sys.argv[1]

ydl_opts = {
    "quiet": True,
    "extract_flat": "in_playlist",
    "skip_download": True,
    "forceurl": True,
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    },
}

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        video_list = info.get("entries", [])
        video_ids = [video.get("id") for video in video_list if video.get("id")]
        print(','.join(video_ids))
except Exception as e:
    print(f"Error fetching video IDs: {e}", file=sys.stderr)
    sys.exit(1)
"""
Direct YouTube Playlist Handler
Provides robust methods for extracting video information from YouTube playlists
without relying on yt-dlp's built-in playlist handling.
"""

import re
import logging
import requests
import json
import os
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class YouTubeDirectHandler:
    """A specialized handler for YouTube playlists that uses multiple methods to extract videos"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def extract_playlist_id(self, playlist_url: str) -> str:
        """Extract the playlist ID from a YouTube URL"""
        pattern = r"(?:list=)([a-zA-Z0-9_-]+)"
        match = re.search(pattern, playlist_url)
        if match:
            return match.group(1)
        # If no match, check if the input might already be just a playlist ID
        if re.match(r"^[a-zA-Z0-9_-]+$", playlist_url):
            return playlist_url
        raise ValueError(f"Could not extract playlist ID from {playlist_url}")

    def get_playlist_videos(self, playlist_url: str) -> List[Dict[str, Any]]:
        """Get a list of videos in a playlist using direct web requests"""
        try:
            logging.info(f"Attempting to extract videos from playlist: {playlist_url}")
            playlist_id = self.extract_playlist_id(playlist_url)
            logging.info(f"Extracted playlist ID: {playlist_id}")

            videos = []
            # Try method 1: Direct API-like request
            try:
                videos = self._method_direct_api(playlist_id)
                if videos:
                    logging.info(
                        f"Successfully extracted {len(videos)} videos using direct API method"
                    )
                    return videos
            except Exception as e:
                logging.warning(f"Direct API method failed: {e}")

            # Try method 2: Web page scraping simulation
            try:
                videos = self._method_web_simulation(playlist_id)
                if videos:
                    logging.info(
                        f"Successfully extracted {len(videos)} videos using web simulation"
                    )
                    return videos
            except Exception as e:
                logging.warning(f"Web simulation method failed: {e}")

            # Try method 3: Individual video fallback (DISABLED)
            try:
                videos = self._method_individual_lookup(playlist_id)
                if videos:
                    logging.info(
                        f"Successfully extracted {len(videos)} videos using individual lookup"
                    )
                    return videos
            except Exception as e:
                logging.warning(f"Individual lookup method failed: {e}")

            if not videos:
                raise ValueError(
                    f"All methods failed to extract videos from playlist {playlist_id}"
                )

            return videos

        except Exception as e:
            logging.error(f"Failed to extract playlist videos: {e}", exc_info=True)
            raise

    def _method_direct_api(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Method 1: Try to use a direct API-like request"""
        logging.info("Trying direct API method...")
        api_url = f"https://www.youtube.com/playlist?list={playlist_id}&hl=en"
        response = requests.get(api_url, headers=self.headers)
        response.raise_for_status()

        # Extract the initial data JSON
        initial_data_match = re.search(r"var ytInitialData = ({.*?});", response.text)
        if not initial_data_match:
            raise ValueError("Could not find ytInitialData in the response")

        initial_data = json.loads(initial_data_match.group(1))

        # Navigate the JSON structure to find video items
        videos = []
        try:
            tabs = initial_data["contents"]["twoColumnBrowseResultsRenderer"]["tabs"]
            playlist_tab = next(
                tab
                for tab in tabs
                if "tabRenderer" in tab and tab["tabRenderer"]["title"] == "Videos"
            )
            video_items = playlist_tab["tabRenderer"]["content"]["sectionListRenderer"][
                "contents"
            ][0]["itemSectionRenderer"]["contents"][0]["playlistVideoListRenderer"][
                "contents"
            ]

            for item in video_items:
                if "playlistVideoRenderer" in item:
                    renderer = item["playlistVideoRenderer"]
                    video_id = renderer["videoId"]
                    title = renderer["title"]["runs"][0]["text"]

                    videos.append(
                        {
                            "id": video_id,
                            "title": title,
                            "url": f"https://www.youtube.com/watch?v={video_id}&list={playlist_id}",
                            "webpage_url": f"https://www.youtube.com/watch?v={video_id}&list={playlist_id}",
                        }
                    )
        except (KeyError, StopIteration) as e:
            logging.warning(f"Error parsing playlist data: {e}")

        return videos

    def _method_web_simulation(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Method 2: Simulate web browser behavior to extract playlist data"""
        logging.info("Trying web simulation method...")
        playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}&hl=en"

        # Add additional headers to better simulate a browser
        enhanced_headers = self.headers.copy()
        enhanced_headers.update(
            {
                "Referer": "https://www.youtube.com/",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            }
        )

        session = requests.Session()
        response = session.get(playlist_url, headers=enhanced_headers)
        response.raise_for_status()

        # Look for video links in the HTML
        video_pattern = r"watch\?v=([a-zA-Z0-9_-]+)(?:&amp;|&)list="
        video_ids = re.findall(video_pattern, response.text)

        # Remove duplicates while preserving order
        unique_ids = []
        for video_id in video_ids:
            if video_id not in unique_ids:
                unique_ids.append(video_id)

        videos = []
        for video_id in unique_ids:
            # Try to extract title
            title_pattern = rf'title="(.*?)"[^>]*?href="/watch\?v={video_id}'
            title_match = re.search(title_pattern, response.text)
            title = title_match.group(1) if title_match else f"Video {video_id}"

            videos.append(
                {
                    "id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}&list={playlist_id}",
                    "webpage_url": f"https://www.youtube.com/watch?v={video_id}&list={playlist_id}",
                }
            )

        return videos

    def _method_individual_lookup(self, playlist_id: str) -> List[Dict[str, Any]]:
        """Method 3: Disabled fallback to avoid invalid video IDs causing download failures"""
        logging.info(
            "Individual video lookup method is disabled to prevent invalid video URLs."
        )
        return []


# Direct use example
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        playlist_url = sys.argv[1]
        handler = YouTubeDirectHandler()
        videos = handler.get_playlist_videos(playlist_url)
        for i, video in enumerate(videos):
            print(f"{i+1}. {video['title']} (ID: {video['id']})")
        print(f"Total videos found: {len(videos)}")
    else:
        print("Usage: python youtube_direct.py <playlist_url>")

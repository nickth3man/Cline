def add_transcript(self, transcript_data):
    """Adds a transcript record to the database."""
    # Existing code...
    logging.info(f"Transcript added for video ID: {transcript_data['video_id']}")
    return True


import sqlite3
import logging
import os
import datetime
import csv
import json
from dataclasses import dataclass, fields
from typing import Optional, List, Type, TypeVar, Dict, Any, Tuple, Iterator

# --- Logging Setup ---
# Use the same logger instance if configured globally, or configure locally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s",
)

DEFAULT_DB_NAME = "pipeline_output.db"

# --- Data Classes ---
T = TypeVar("T")  # Generic type variable for dataclasses


@dataclass
class VideoDetails:
    """Represents detailed information for a video entry."""

    video_id: str  # YouTube video ID (Primary Key in DB)
    title: Optional[str] = None
    processing_status: Optional[str] = "pending"
    folder_path: Optional[str] = None  # Path relative to work_dir
    playlist_url: Optional[str] = None
    sanitized_title: Optional[str] = None
    upload_date: Optional[str] = None
    duration: Optional[int] = None
    channel: Optional[str] = None
    description: Optional[str] = None
    metadata_json_path: Optional[str] = None
    audio_wav_path: Optional[str] = None
    last_updated: Optional[str] = None  # Timestamp as string

    # Allow creating from a dictionary, ignoring extra keys
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


@dataclass
class VideoPaths:
    """Represents file paths associated with a video."""

    video_id: str
    folder_path: Optional[str] = None
    metadata_json_path: Optional[str] = None
    audio_wav_path: Optional[str] = None
    corrected_transcript_path: Optional[str] = None
    summary_path: Optional[str] = None  # Assuming summaries might have a path
    error_log_path: Optional[str] = None  # Derived path


@dataclass
class TranscriptDetails:
    """Represents details of a stored transcript."""

    transcript_id: int
    video_id: str
    transcript_type: Optional[str] = "corrected"
    model_used: Optional[str] = None
    file_path: Optional[str] = None
    content: Optional[str] = None
    created_at: Optional[str] = None


# --- Database Manager Class ---


class DatabaseManager:
    """Manages interactions with the SQLite database for the pipeline."""

    def __init__(self, db_path=DEFAULT_DB_NAME):
        self.db_path = db_path
        logging.info(f"DatabaseManager initialized for path: {self.db_path}")
        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logging.info(f"Created directory for database: {db_dir}")
        self.initialize_database()  # Ensure schema exists on instantiation

    def _get_connection(self):
        """Establishes a connection to the SQLite database."""
        logging.debug(f"Connecting to database at: {self.db_path}")
        try:
            conn = sqlite3.connect(self.db_path)
            # No row_factory needed here, mapping happens in _execute_select
            conn.execute("PRAGMA foreign_keys = ON;")  # Enforce foreign key constraints
            logging.debug("Database connection successful.")
            return conn
        except sqlite3.Error as e:
            logging.error(
                f"Error connecting to database {self.db_path}: {e}", exc_info=True
            )
            raise

    def _execute_select(
        self, sql: str, params: tuple = (), dataclass_type: Optional[Type[T]] = None
    ) -> List[T | tuple]:
        """Executes a SELECT query and returns results, optionally mapped to a dataclass."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            results = cursor.fetchall()  # List of tuples

            if dataclass_type and results:
                column_names = [description[0] for description in cursor.description]
                try:
                    # Map tuples to dataclass instances
                    return [
                        dataclass_type(**dict(zip(column_names, row)))
                        for row in results
                    ]
                except TypeError as e:
                    logging.error(
                        f"Error mapping query results to dataclass {dataclass_type.__name__}: {e}",
                        exc_info=True,
                    )
                    logging.error(f"Column names: {column_names}")
                    logging.error(f"First row data: {results[0] if results else 'N/A'}")
                    raise  # Re-raise after logging details
            else:
                return results  # Return list of tuples if no dataclass specified or no results
        except sqlite3.Error as e:
            logging.error(
                f"Error executing SELECT query: {e}\nSQL: {sql}\nParams: {params}",
                exc_info=True,
            )
            raise
        finally:
            if conn:
                conn.close()
                logging.debug("Database connection closed after SELECT.")

    def _execute_update(self, sql: str, params: tuple = ()) -> int:
        """Executes an INSERT, UPDATE, or DELETE query and returns row count."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            rowcount = cursor.rowcount
            logging.debug(
                f"Executed update. Rows affected: {rowcount}. SQL: {sql[:100]}..."
            )
            return rowcount
        except sqlite3.Error as e:
            logging.error(
                f"Error executing UPDATE/INSERT/DELETE query: {e}\nSQL: {sql}\nParams: {params}",
                exc_info=True,
            )
            if conn:
                conn.rollback()
                logging.warning("Transaction rolled back due to error.")
            raise
        finally:
            if conn:
                conn.close()
                logging.debug("Database connection closed after UPDATE/INSERT/DELETE.")

    # --- Export Methods ---

    def _build_export_query(
        self, video_ids: Optional[List[str]] = None, status: Optional[str] = None
    ) -> Tuple[str, tuple]:
        """Builds the SQL query and parameters for exporting video data."""
        select_columns = """
            SELECT
                v.video_id, v.title, v.processing_status, v.folder_path, v.playlist_url,
                v.sanitized_title, v.upload_date, v.duration, v.channel, v.description,
                v.metadata_json_path, v.audio_wav_path, v.last_updated,
                t.content as transcript_content, t.model_used as transcript_model,
                s.content as summary_content, s.model_used as summary_model
            FROM videos v
            LEFT JOIN transcripts t ON v.video_id = t.video_id AND t.transcript_type = 'corrected' -- Assuming latest/only corrected
            LEFT JOIN summaries s ON v.video_id = s.video_id -- Assuming latest/only summary
        """
        where_clauses = []
        params = []

        if video_ids:
            # Ensure video_ids are strings if they aren't already (e.g., from CLI int parsing)
            str_video_ids = [str(vid) for vid in video_ids]
            placeholders = ", ".join("?" * len(str_video_ids))
            where_clauses.append(f"v.video_id IN ({placeholders})")
            params.extend(str_video_ids)

        if status:
            # Use LIKE for potential wildcard matching (e.g., 'error_%')
            where_clauses.append("v.processing_status LIKE ?")
            params.append(status)

        sql = select_columns
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        sql += " ORDER BY v.last_updated DESC;"  # Consistent ordering

        logging.debug(f"Built export query: {sql[:200]}... with params: {params}")
        return sql, tuple(params)

    def _stream_query_results(
        self, sql: str, params: tuple = (), chunk_size: int = 100
    ) -> Iterator[List[tuple]]:
        """Executes a SELECT query and yields results in chunks."""
        conn = self._get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            while True:
                results = cursor.fetchmany(chunk_size)
                if not results:
                    break
                yield results
        except sqlite3.Error as e:
            logging.error(
                f"Error streaming query results: {e}\nSQL: {sql}\nParams: {params}",
                exc_info=True,
            )
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                logging.debug("Database connection closed after streaming.")

    def export_to_csv(
        self,
        filepath: str,
        video_ids: Optional[List[str]] = None,
        status: Optional[str] = None,
        chunk_size: int = 100,
    ) -> int:
        """Exports video data (optionally filtered) to a CSV file using streaming."""
        sql, params = self._build_export_query(video_ids, status)
        rows_exported = 0
        is_header_written = False

        try:
            # Ensure directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Created directory for export file: {dir_path}")

            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)

                conn = (
                    self._get_connection()
                )  # Need a separate connection for getting headers
                cursor = None
                try:
                    cursor = conn.cursor()
                    cursor.execute(sql, params)  # Execute once to get description
                    headers = [description[0] for description in cursor.description]
                    csv_writer.writerow(headers)
                    is_header_written = True
                finally:
                    if cursor:
                        cursor.close()
                    if conn:
                        conn.close()

                # Now stream results using the generator
                for chunk in self._stream_query_results(sql, params, chunk_size):
                    csv_writer.writerows(chunk)
                    rows_exported += len(chunk)
                    logging.debug(
                        f"Exported chunk of {len(chunk)} rows to CSV. Total: {rows_exported}"
                    )

            logging.info(
                f"Successfully exported {rows_exported} rows to CSV: {filepath}"
            )
            return rows_exported

        except (sqlite3.Error, IOError, OSError) as e:
            logging.error(
                f"Error exporting data to CSV '{filepath}': {e}", exc_info=True
            )
            # Attempt to remove partially written file on error
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logging.warning(
                        f"Removed partially written CSV file due to error: {filepath}"
                    )
                except OSError as rm_err:
                    logging.error(
                        f"Could not remove partial CSV file '{filepath}': {rm_err}"
                    )
            raise  # Re-raise the original error

    def export_to_json(
        self,
        filepath: str,
        video_ids: Optional[List[str]] = None,
        status: Optional[str] = None,
        chunk_size: int = 100,
    ) -> int:
        """Exports video data (optionally filtered) to a JSON Lines file using streaming."""
        sql, params = self._build_export_query(video_ids, status)
        rows_exported = 0
        headers = []

        try:
            # Ensure directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
                logging.info(f"Created directory for export file: {dir_path}")

            # Get headers first
            conn = self._get_connection()
            cursor = None
            try:
                cursor = conn.cursor()
                cursor.execute(sql, params)  # Execute once to get description
                headers = [description[0] for description in cursor.description]
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()

            if not headers:
                logging.warning(
                    "No headers found for JSON export query, cannot proceed."
                )
                return 0  # Or raise error

            with open(filepath, "w", encoding="utf-8") as jsonfile:
                # Stream results using the generator
                for chunk in self._stream_query_results(sql, params, chunk_size):
                    for row in chunk:
                        row_dict = dict(zip(headers, row))
                        json_line = json.dumps(row_dict, ensure_ascii=False)
                        jsonfile.write(json_line + "\n")
                        rows_exported += 1
                    logging.debug(
                        f"Exported chunk to JSON Lines. Total: {rows_exported}"
                    )

            logging.info(
                f"Successfully exported {rows_exported} rows to JSON Lines: {filepath}"
            )
            return rows_exported

        except (sqlite3.Error, IOError, OSError, json.JSONDecodeError) as e:
            logging.error(
                f"Error exporting data to JSON Lines '{filepath}': {e}", exc_info=True
            )
            # Attempt to remove partially written file on error
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logging.warning(
                        f"Removed partially written JSON Lines file due to error: {filepath}"
                    )
                except OSError as rm_err:
                    logging.error(
                        f"Could not remove partial JSON Lines file '{filepath}': {rm_err}"
                    )
            raise  # Re-raise the original error

    def initialize_database(self):
        """Creates the database and necessary tables if they don't exist."""
        logging.info(f"Initializing database schema in: {self.db_path}")
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # --- Create videos Table ---
            # Added last_updated trigger
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                playlist_url TEXT,
                title TEXT,
                sanitized_title TEXT UNIQUE,
                upload_date TEXT,
                duration INTEGER,
                channel TEXT,
                description TEXT,
                folder_path TEXT UNIQUE NOT NULL,
                metadata_json_path TEXT,
                audio_wav_path TEXT,
                processing_status TEXT DEFAULT 'pending',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            )
            logging.debug("Table 'videos' checked/created.")

            # Trigger to update last_updated timestamp
            cursor.execute(
                """
            CREATE TRIGGER IF NOT EXISTS update_videos_last_updated
            AFTER UPDATE ON videos
            FOR EACH ROW
            BEGIN
                UPDATE videos SET last_updated = CURRENT_TIMESTAMP WHERE video_id = OLD.video_id;
            END;
            """
            )
            logging.debug("Trigger 'update_videos_last_updated' checked/created.")

            # --- Create transcripts Table ---
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS transcripts (
                transcript_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                transcript_type TEXT DEFAULT 'corrected',
                model_used TEXT,
                file_path TEXT UNIQUE,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (video_id) ON DELETE CASCADE
            );
            """
            )
            logging.debug("Table 'transcripts' checked/created.")

            # --- Create summaries Table ---
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT NOT NULL,
                model_used TEXT,
                file_path TEXT UNIQUE,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos (video_id) ON DELETE CASCADE
            );
            """
            )
            logging.debug("Table 'summaries' checked/created.")

            # --- Create Indexes ---
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_videos_playlist ON videos (playlist_url);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_videos_status ON videos (processing_status);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts (video_id);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_summaries_video_id ON summaries (video_id);"
            )
            logging.debug("Indexes checked/created.")

            conn.commit()
            logging.info("Database schema initialized successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error initializing database schema: {e}", exc_info=True)
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
                logging.debug("Database connection closed after initialization.")

    def add_or_update_video(self, video_data: Dict[str, Any]):
        """
        Adds a new video record or updates an existing one based on video_id.
        video_data should be a dictionary. Uses VideoDetails.from_dict for filtering.
        """
        required_keys = {"video_id", "folder_path"}
        if not required_keys.issubset(video_data.keys()):
            raise ValueError(
                f"Missing required keys for video record: {required_keys - set(video_data.keys())}"
            )

        # Use dataclass to filter and potentially set defaults (though defaults are mainly in SQL)
        video_obj = VideoDetails.from_dict(video_data)

        sql = """
        INSERT INTO videos (
            video_id, playlist_url, title, sanitized_title, upload_date, duration,
            channel, description, folder_path, metadata_json_path, audio_wav_path,
            processing_status, last_updated
        ) VALUES (:video_id, :playlist_url, :title, :sanitized_title, :upload_date, :duration,
                  :channel, :description, :folder_path, :metadata_json_path, :audio_wav_path,
                  :processing_status, CURRENT_TIMESTAMP)
        ON CONFLICT(video_id) DO UPDATE SET
            playlist_url=excluded.playlist_url,
            title=excluded.title,
            sanitized_title=excluded.sanitized_title,
            upload_date=excluded.upload_date,
            duration=excluded.duration,
            channel=excluded.channel,
            description=excluded.description,
            folder_path=excluded.folder_path,
            metadata_json_path=excluded.metadata_json_path,
            audio_wav_path=excluded.audio_wav_path,
            processing_status=excluded.processing_status,
            last_updated=CURRENT_TIMESTAMP;
        """
        # Convert dataclass to dict for named parameters
        params_dict = video_obj.__dict__
        try:
            self._execute_update(sql, params_dict)  # Use dict for named placeholders
            logging.info(f"Video record added/updated for ID: {video_obj.video_id}")
            return True
        except sqlite3.Error:
            # Error already logged in _execute_update
            return False

    def add_transcript(self, transcript_data: Dict[str, Any]) -> Optional[int]:
        """
        Adds a transcript record.
        transcript_data should be a dictionary.
        """
        required_keys = {"video_id", "content"}
        if not required_keys.issubset(transcript_data.keys()):
            raise ValueError(
                f"Missing required keys for transcript record: {required_keys - set(transcript_data.keys())}"
            )

        sql = """
        INSERT INTO transcripts (
            video_id, transcript_type, model_used, file_path, content, created_at
        ) VALUES (:video_id, :transcript_type, :model_used, :file_path, :content, CURRENT_TIMESTAMP);
        """
        # Prepare params dict, setting defaults if necessary
        params = {
            "video_id": transcript_data.get("video_id"),
            "transcript_type": transcript_data.get("transcript_type", "corrected"),
            "model_used": transcript_data.get("model_used"),
            "file_path": transcript_data.get("file_path"),
            "content": transcript_data.get("content"),
        }

        conn = self._get_connection()  # Need connection for lastrowid
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            last_id = cursor.lastrowid
            conn.commit()
            logging.info(
                f"Transcript record added for video ID: {params['video_id']}, transcript_id: {last_id}"
            )
            return last_id
        except sqlite3.Error as e:
            logging.error(
                f"Error adding transcript for video {params.get('video_id', 'N/A')}: {e}",
                exc_info=True,
            )
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()

    # --- New Management Methods ---

    def get_all_video_details(self) -> List[VideoDetails]:
        """Fetches key details for all videos, ordered by last updated."""
        # Select columns matching VideoDetails fields
        sql = """
            SELECT
                video_id, title, processing_status, folder_path, playlist_url,
                sanitized_title, upload_date, duration, channel, description,
                metadata_json_path, audio_wav_path, last_updated
            FROM videos
            ORDER BY last_updated DESC;
        """
        return self._execute_select(sql, dataclass_type=VideoDetails)

    def get_videos_by_status(self, status_pattern: str) -> List[VideoDetails]:
        """Fetches videos matching a status pattern (e.g., 'error_%')."""
        # Select columns matching VideoDetails fields
        sql = """
            SELECT
                video_id, title, processing_status, folder_path, playlist_url,
                sanitized_title, upload_date, duration, channel, description,
                metadata_json_path, audio_wav_path, last_updated
            FROM videos
            WHERE processing_status LIKE ?
            ORDER BY last_updated DESC;
        """
        return self._execute_select(sql, (status_pattern,), dataclass_type=VideoDetails)

    def get_video_details(self, video_id: str) -> Optional[VideoDetails]:
        """Fetches all details for a single video by its YouTube video_id."""
        # Select columns matching VideoDetails fields
        sql = """
            SELECT
                video_id, title, processing_status, folder_path, playlist_url,
                sanitized_title, upload_date, duration, channel, description,
                metadata_json_path, audio_wav_path, last_updated
            FROM videos
            WHERE video_id = ?;
        """
        results = self._execute_select(sql, (video_id,), dataclass_type=VideoDetails)
        return results[0] if results else None

    def get_video_paths(self, video_id: str) -> Optional[VideoPaths]:
        """Retrieves known file paths for a video by its YouTube video_id."""
        # Fetch paths stored in DB
        sql = """
            SELECT
                v.video_id, v.folder_path, v.metadata_json_path, v.audio_wav_path,
                t.file_path as corrected_transcript_path,
                s.file_path as summary_path
            FROM videos v
            LEFT JOIN transcripts t ON v.video_id = t.video_id AND t.transcript_type = 'corrected'
            LEFT JOIN summaries s ON v.video_id = s.video_id
            WHERE v.video_id = ?;
        """
        results = self._execute_select(sql, (video_id,))  # Get raw tuple first

        if not results:
            return None

        # Map raw tuple result to VideoPaths dataclass
        row = results[0]
        col_names = [
            "video_id",
            "folder_path",
            "metadata_json_path",
            "audio_wav_path",
            "corrected_transcript_path",
            "summary_path",
        ]
        paths_dict = dict(zip(col_names, row))

        # Derive error log path
        error_log_path = None
        if paths_dict.get("folder_path"):
            # Assuming error log is always named 'error.log' within the folder
            error_log_path = os.path.join(paths_dict["folder_path"], "error.log")

        return VideoPaths(
            video_id=paths_dict.get("video_id"),
            folder_path=paths_dict.get("folder_path"),
            metadata_json_path=paths_dict.get("metadata_json_path"),
            audio_wav_path=paths_dict.get("audio_wav_path"),
            corrected_transcript_path=paths_dict.get("corrected_transcript_path"),
            summary_path=paths_dict.get("summary_path"),
            error_log_path=error_log_path,
        )

    def update_video_status(self, video_id: str, new_status: str) -> bool:
        """Updates the processing_status of a specific video."""
        # last_updated is handled by the trigger
        sql = "UPDATE videos SET processing_status = ? WHERE video_id = ?;"
        rows_affected = self._execute_update(sql, (new_status, video_id))
        if rows_affected > 0:
            logging.info(f"Updated status for video {video_id} to '{new_status}'")
        else:
            logging.warning(
                f"Attempted to update status for non-existent video ID: {video_id}"
            )
        return rows_affected > 0

    def delete_video(self, video_id: str) -> Optional[VideoPaths]:
        """
        Deletes video record and associated transcript/summary records.
        Returns the associated paths if deletion was successful, otherwise None.
        NOTE: This only deletes DB records, not files on disk.
        """
        paths = self.get_video_paths(video_id)  # Get paths before deleting
        if not paths:
            logging.warning(f"Attempted to delete non-existent video ID: {video_id}")
            return None  # Video not found

        # Foreign key constraints with ON DELETE CASCADE should handle transcript/summary deletion
        sql = "DELETE FROM videos WHERE video_id = ?;"
        try:
            rows_affected = self._execute_update(sql, (video_id,))
            if rows_affected > 0:
                logging.info(f"Deleted database records for video ID: {video_id}")
                return paths
            else:
                # Should not happen if paths were found, but handle defensively
                logging.warning(
                    f"Video ID {video_id} found by get_video_paths but delete affected 0 rows."
                )
                return None
        except sqlite3.Error:
            # Error logged in _execute_update
            return None

    def add_summary(self, summary_data: Dict[str, Any]) -> Optional[int]:
        """
        Adds a summary record.
        summary_data should be a dictionary.
        """
        required_keys = {"video_id", "content"}
        if not required_keys.issubset(summary_data.keys()):
            raise ValueError(
                f"Missing required keys for summary record: {required_keys - set(summary_data.keys())}"
            )

        sql = """
        INSERT INTO summaries (
            video_id, model_used, file_path, content, created_at
        ) VALUES (:video_id, :model_used, :file_path, :content, CURRENT_TIMESTAMP);
        """
        # Prepare params dict, setting defaults if necessary
        params = {
            "video_id": summary_data.get("video_id"),
            "model_used": summary_data.get("model_used"),
            "file_path": summary_data.get("file_path"),
            "content": summary_data.get("content"),
        }

        conn = self._get_connection()  # Need connection for lastrowid
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            last_id = cursor.lastrowid
            conn.commit()
            logging.info(
                f"Summary record added for video ID: {params['video_id']}, summary_id: {last_id}"
            )
            return last_id
        except sqlite3.Error as e:
            logging.error(
                f"Error adding summary for video {params.get('video_id', 'N/A')}: {e}",
                exc_info=True,
            )
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                conn.close()


# --- Example Usage ---
if __name__ == "__main__":
    # Example usage: Initialize the database in the current directory
    db_file = "test_pipeline_managed.db"
    if os.path.exists(db_file):
        try:
            os.remove(db_file)  # Start fresh for testing
            logging.info(f"Removed existing test database: {db_file}")
        except OSError as e:
            logging.error(f"Error removing existing test database {db_file}: {e}")

    logging.info("Running DatabaseManager initialization example...")
    try:
        manager = DatabaseManager(db_file)
        logging.info("Initialization example complete.")

        # Example of adding data
        logging.info("Running data insertion example...")
        test_video_data = {
            "video_id": "dQw4w9WgXcQ",
            "playlist_url": "test_playlist",
            "title": "Rick Astley - Never Gonna Give You Up (Official Music Video)",
            "sanitized_title": "Rick_Astley_Never_Gonna_Give_You_Up_Official_Music_Video",
            "upload_date": "20091025",
            "duration": 212,
            "channel": "Rick Astley",
            "description": "The official video for “Never Gonna Give You Up” by Rick Astley",
            "folder_path": "yt_pipeline_output/Rick_Astley_Never_Gonna_Give_You_Up_Official_Music_Video",
            "metadata_json_path": "yt_pipeline_output/Rick_Astley_Never_Gonna_Give_You_Up_Official_Music_Video/video.info.json",
            "audio_wav_path": "yt_pipeline_output/Rick_Astley_Never_Gonna_Give_You_Up_Official_Music_Video/audio.wav",
            "processing_status": "downloaded",
        }
        if manager.add_or_update_video(test_video_data):
            test_transcript_data = {
                "video_id": "dQw4w9WgXcQ",
                "transcript_type": "corrected",
                "model_used": "claude-3-haiku",
                "file_path": "yt_pipeline_output/Rick_Astley_Never_Gonna_Give_You_Up_Official_Music_Video/corrected_transcript.md",
                "content": "We're no strangers to love...",
            }
            manager.add_transcript(test_transcript_data)

        logging.info("Data insertion example complete.")

        # Example of using new methods
        logging.info("Running data retrieval examples...")
        all_videos = manager.get_all_video_details()
        logging.info(
            f"Retrieved {len(all_videos)} video(s). First one: {all_videos[0] if all_videos else 'None'}"
        )

        details = manager.get_video_details("dQw4w9WgXcQ")
        logging.info(f"Details for dQw4w9WgXcQ: {details}")

        paths = manager.get_video_paths("dQw4w9WgXcQ")
        logging.info(f"Paths for dQw4w9WgXcQ: {paths}")

        # Example update status
        manager.update_video_status("dQw4w9WgXcQ", "processed")
        details_after_update = manager.get_video_details("dQw4w9WgXcQ")
        logging.info(f"Details after status update: {details_after_update}")

        # Example delete (only DB records)
        # deleted_paths = manager.delete_video('dQw4w9WgXcQ')
        # logging.info(f"Attempted delete. Paths returned: {deleted_paths}")
        # details_after_delete = manager.get_video_details('dQw4w9WgXcQ')
        # logging.info(f"Details after delete attempt: {details_after_delete}")

        # Example export
        logging.info("Running data export examples...")
        try:
            csv_count = manager.export_to_csv("test_export.csv", status="processed")
            logging.info(f"CSV export example complete. Exported {csv_count} rows.")
            json_count = manager.export_to_json("test_export.jsonl")
            logging.info(
                f"JSON Lines export example complete. Exported {json_count} rows."
            )
            # Example filtered export
            filtered_csv_count = manager.export_to_csv(
                "test_export_filtered.csv", video_ids=["dQw4w9WgXcQ"]
            )
            logging.info(
                f"Filtered CSV export example complete. Exported {filtered_csv_count} rows."
            )

        except Exception as export_err:
            logging.error(f"Error during export example: {export_err}", exc_info=True)

    except Exception as e:
        logging.exception("An error occurred during the DatabaseManager example usage.")
    finally:
        # Clean up test files
        for f in [
            "test_pipeline_managed.db",
            "test_export.csv",
            "test_export.jsonl",
            "test_export_filtered.csv",
        ]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    logging.info(f"Cleaned up test file: {f}")
                except OSError as e:
                    logging.error(f"Error cleaning up test file {f}: {e}")

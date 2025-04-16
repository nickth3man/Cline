import sys
import threading
import os
import traceback
import logging
import re
import json
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QMessageBox,
    QFileDialog,
    QGridLayout,
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from dotenv import load_dotenv
import yt_dlp

# Import the refactored workflow logic
import workflow_logic

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
)
# Load .env file for API keys if it exists
load_dotenv()


def sanitize_filename(filename):
    """Removes invalid characters for filenames/paths."""
    sanitized = re.sub(r"[^\w\-. ]", "", filename)  # Allow dots and hyphens
    sanitized = re.sub(
        r"\s+", "_", sanitized
    )  # Replace whitespace sequences with single underscore
    return sanitized[:150]  # Limit length further


class ProcessingThread(QThread):
    # Signals for UI updates
    status_update = pyqtSignal(str)
    video_status_update = pyqtSignal(int, str, str)  # video_index, status, details
    error_signal = pyqtSignal(int, str)  # video_index (-1 for general), error_message
    progress_update = pyqtSignal(int, int)  # current_video_count, total_videos
    finished_signal = pyqtSignal()  # Signal emitted when run() completes

    def __init__(self, playlist_url, resolution, output_path, correction_model):
        super().__init__()
        self.playlist_url = playlist_url
        self.resolution = resolution
        self.output_path = output_path
        self.correction_model = correction_model
        self.video_list = []
        self._is_running = True
        self.current_operation_lock = threading.Lock()
        self.current_operation = ""
        logging.info("ProcessingThread initialized.")

    def run(self):
        logging.info("ProcessingThread started.")
        total_videos = 0
        processed_count = 0

        try:
            # --- Fetch Playlist Info ---
            if not self._is_running:
                return
            self.set_operation("Fetching playlist info...")
            self.status_update.emit(self.current_operation)
            ydl_opts_info = {
                "quiet": True,
                "extract_flat": "in_playlist",
                "skip_download": True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
                    logging.info(f"Extracting info for playlist: {self.playlist_url}")
                    info = ydl.extract_info(self.playlist_url, download=False)
                    if "entries" in info and info["entries"]:
                        self.video_list = info["entries"]
                        total_videos = len(self.video_list)
                        logging.info(f"Fetched {total_videos} videos.")
                        self.status_update.emit(
                            f"Fetched {total_videos} videos. Starting processing..."
                        )
                        self.progress_update.emit(0, total_videos)
                    else:
                        raise ValueError(
                            "No videos found in the playlist or playlist is invalid."
                        )
            except Exception as e:
                error_msg = f"Failed to extract playlist info: {str(e)}"
                logging.error(f"{error_msg}\n{traceback.format_exc()}")
                self.error_signal.emit(-1, error_msg)
                return
            finally:
                self.clear_operation()

            # --- Process Each Video ---
            for index, video_entry in enumerate(self.video_list):
                if not self._is_running:
                    logging.info("Processing stopped by user request.")
                    break

                processed_count = index
                video_title = video_entry.get("title", f"video_{index}")
                video_url = video_entry.get("webpage_url", video_entry.get("id"))
                sanitized_title = sanitize_filename(video_title)
                video_folder_path = os.path.join(
                    self.output_path, f"{index:03d}_{sanitized_title}"
                )
                download_path = os.path.join(
                    video_folder_path, f"{sanitized_title}.mp4"
                )
                wav_audio_path = os.path.join(
                    video_folder_path, f"{sanitized_title}_audio.wav"
                )
                raw_transcript_path = os.path.join(
                    video_folder_path, f"{sanitized_title}_raw_transcript.txt"
                )
                diarization_path = os.path.join(
                    video_folder_path, f"{sanitized_title}_diarization.json"
                )
                corrected_transcript_path = os.path.join(
                    video_folder_path, f"{sanitized_title}_corrected.md"
                )

                logging.info(
                    f"--- Processing video {index + 1}/{total_videos}: {video_title} ---"
                )
                self.video_status_update.emit(index, "Preparing", video_title)

                try:
                    # Create video-specific directory
                    if not self._is_running:
                        break
                    self.set_operation(f"Creating directory for {video_title}")
                    os.makedirs(video_folder_path, exist_ok=True)
                    logging.info(f"Ensured directory exists: {video_folder_path}")
                    self.clear_operation()

                    # --- 1. Download Video ---
                    if not self._is_running:
                        break
                    self.set_operation(f"Downloading {video_title}")
                    self.video_status_update.emit(index, "Downloading", "")
                    try:
                        ydl_opts_download = {
                            "format": f"bestvideo[height<={self.resolution}][ext=mp4]+bestaudio[ext=m4a]/best[height<={self.resolution}][ext=mp4]/best",
                            "outtmpl": download_path,
                            "quiet": True,
                            "noprogress": True,
                            "postprocessors": [
                                {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}
                            ],
                            "http_headers": {"User-Agent": "Mozilla/5.0"},
                        }
                        logging.info(
                            f"Starting download for: {video_url} to {download_path}"
                        )
                        with yt_dlp.YoutubeDL(ydl_opts_download) as ydl_download:
                            ydl_download.download([video_url])
                        logging.info(f"Download complete: {download_path}")
                    except Exception as e:
                        raise RuntimeError(f"Download failed: {str(e)}")
                    finally:
                        self.clear_operation()

                    # --- 2. Extract/Convert Audio ---
                    if not self._is_running:
                        break
                    self.set_operation(f"Converting audio for {video_title}")
                    self.video_status_update.emit(index, "Converting Audio", "")
                    try:
                        workflow_logic.extract_or_convert_audio(
                            download_path, wav_audio_path
                        )
                    except Exception as e:
                        raise RuntimeError(f"Audio conversion failed: {str(e)}")
                    finally:
                        self.clear_operation()

                    # --- 3. Transcribe Audio ---
                    if not self._is_running:
                        break
                    self.set_operation(f"Transcribing {video_title}")
                    self.video_status_update.emit(index, "Transcribing", "")
                    try:
                        raw_transcript = workflow_logic.transcribe_audio(wav_audio_path)
                        with open(raw_transcript_path, "w", encoding="utf-8") as f:
                            f.write(raw_transcript)
                        logging.info(f"Raw transcript saved: {raw_transcript_path}")
                    except Exception as e:
                        raise RuntimeError(f"Transcription failed: {str(e)}")
                    finally:
                        self.clear_operation()

                    # --- 4. Diarize Speakers ---
                    if not self._is_running:
                        break
                    if workflow_logic.diarization_pipeline:
                        self.set_operation(f"Diarizing speakers for {video_title}")
                        self.video_status_update.emit(index, "Diarizing", "")
                        try:
                            diarization_result = workflow_logic.diarize_speakers(
                                wav_audio_path
                            )
                            with open(diarization_path, "w", encoding="utf-8") as f:
                                json.dump(diarization_result, f, indent=2)
                            logging.info(
                                f"Diarization results saved: {diarization_path}"
                            )
                        except Exception as e:
                            logging.error(
                                f"Diarization step failed for {video_title}: {str(e)}",
                                exc_info=True,
                            )
                            self.error_signal.emit(
                                index, f"Diarization Failed: {str(e)}"
                            )
                            self.video_status_update.emit(
                                index, "Warning", "Diarization Failed"
                            )
                            diarization_result = []
                        finally:
                            self.clear_operation()
                    else:
                        logging.warning(
                            f"Skipping diarization for {video_title} as pipeline is not available."
                        )
                        self.video_status_update.emit(index, "Skipping Diarization", "")
                        diarization_result = []

                    # --- 5. Correct Transcript ---
                    if not self._is_running:
                        break
                    self.set_operation(f"Correcting transcript for {video_title}")
                    self.video_status_update.emit(index, "Correcting", "")
                    try:
                        corrected_transcript = workflow_logic.correct_transcript(
                            raw_transcript,
                            diarization_result,
                            correction_model=self.correction_model,
                        )
                        with open(
                            corrected_transcript_path, "w", encoding="utf-8"
                        ) as f:
                            f.write(corrected_transcript)
                        logging.info(
                            f"Corrected transcript saved: {corrected_transcript_path}"
                        )
                    except Exception as e:
                        raise RuntimeError(f"Transcript correction failed: {str(e)}")
                    finally:
                        self.clear_operation()

                    # --- Mark as Complete ---
                    self.video_status_update.emit(
                        index, "Complete", corrected_transcript_path
                    )
                    processed_count = index + 1

                except Exception as e:
                    error_msg = (
                        f"Failed processing video {index} ({video_title}): {str(e)}"
                    )
                    logging.error(f"{error_msg}\n{traceback.format_exc()}")
                    self.error_signal.emit(index, error_msg)
                    self.video_status_update.emit(index, "Error", str(e)[:100])
                    # Continue to the next video

                finally:
                    self.clear_operation()
                    self.progress_update.emit(processed_count, total_videos)

            # --- End of Playlist ---
            if self._is_running:
                logging.info("Playlist processing finished.")
                self.status_update.emit("Playlist processing complete.")
            else:
                self.progress_update.emit(processed_count, total_videos)

        except Exception as e:
            error_msg = f"An critical error occurred in the processing thread: {str(e)}"
            logging.critical(f"{error_msg}\n{traceback.format_exc()}")
            self.error_signal.emit(-1, error_msg)
            self.status_update.emit("Critical error occurred. See logs.")
        finally:
            self.clear_operation()
            self.finished_signal.emit()

    def set_operation(self, description):
        """Safely set the current long-running operation description."""
        with self.current_operation_lock:
            self.current_operation = description
            logging.debug(f"Current operation set: {description}")

    def clear_operation(self):
        """Safely clear the current operation description."""
        with self.current_operation_lock:
            self.current_operation = ""

    def stop(self):
        """Signals the thread to stop gracefully."""
        logging.info("Stop signal received by ProcessingThread.")
        self._is_running = False
        with self.current_operation_lock:
            op = self.current_operation
        if op:
            logging.info(f"Attempting to stop during operation: {op}")
        else:
            logging.info("Stop requested between operations.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Playlist Advanced Processor")
        self.setGeometry(100, 100, 950, 750)
        logging.info("Initializing MainWindow.")

        self.thread = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Input Area ---
        input_group = QWidget()
        input_layout = QGridLayout(input_group)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube Playlist URL")
        input_layout.addWidget(QLabel("Playlist URL:"), 0, 0)
        input_layout.addWidget(self.url_input, 0, 1, 1, 3)

        self.resolution_input = QLineEdit("720")
        self.resolution_input.setPlaceholderText("e.g., 1080, 720, best")
        input_layout.addWidget(QLabel("Max Video Res:"), 1, 0)
        input_layout.addWidget(self.resolution_input, 1, 1)

        # Updated: Model for Correction
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            [
                "anthropic/claude-3-haiku-20240307",
                "anthropic/claude-3-sonnet-20240229",
                "openai/gpt-4-turbo",
                "google/gemini-pro",
                "mistralai/mistral-large-latest",
            ]
        )
        input_layout.addWidget(QLabel("Correction Model:"), 1, 2)
        input_layout.addWidget(self.model_combo, 1, 3)

        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select Output Directory")
        self.output_button = QPushButton("Browse...")
        self.output_button.clicked.connect(self.browse_output_directory)
        input_layout.addWidget(QLabel("Output Path:"), 2, 0)
        input_layout.addWidget(self.output_input, 2, 1, 1, 2)
        input_layout.addWidget(self.output_button, 2, 3)

        main_layout.addWidget(input_group)

        # --- Control Buttons ---
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)

        # --- Progress & Status ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Waiting to start...")
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel(
            "Ready. Ensure OPENROUTER_API_KEY and HF_TOKEN (optional) are set in .env file or environment."
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # --- Video Table ---
        self.video_table = QTableWidget(0, 3)
        self.video_table.setHorizontalHeaderLabels(
            ["#", "Video Title", "Status/Result"]
        )
        self.video_table.setColumnWidth(0, 40)
        self.video_table.setColumnWidth(1, 400)
        self.video_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.video_table)

        # --- Error Log ---
        main_layout.addWidget(QLabel("Log / Errors:"))
        self.error_text = QTextEdit()
        self.error_text.setReadOnly(True)
        self.error_text.setMaximumHeight(150)
        main_layout.addWidget(self.error_text)

        # Check for FFmpeg on startup
        QTimer.singleShot(100, self.initial_checks)

        logging.info("MainWindow UI setup complete.")

    def initial_checks(self):
        """Perform checks after UI is loaded."""
        if not workflow_logic.check_ffmpeg():
            QMessageBox.warning(
                self,
                "Dependency Check Failed",
                "FFmpeg not found in system PATH. Audio extraction/conversion will fail. "
                "Please install FFmpeg and ensure it's accessible.",
            )
            self.status_label.setText("Error: FFmpeg not found!")
        if not os.getenv("OPENROUTER_API_KEY"):
            QMessageBox.warning(
                self,
                "Configuration Error",
                "OPENROUTER_API_KEY not found in environment variables or .env file. "
                "API calls will fail. Please set it.",
            )
            self.status_label.setText("Error: OPENROUTER_API_KEY not set!")
        if workflow_logic.diarization_pipeline is None:
            logging.warning("Pyannote pipeline failed to load during init.")
            self.error_text.append(
                "Warning: Pyannote diarization pipeline failed to load. Diarization will be skipped. Check logs and HF token/model terms."
            )

    def browse_output_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_input.setText(directory)
            logging.info(f"Output directory selected: {directory}")

    def start_processing(self):
        logging.info("Start processing button clicked.")
        playlist_url = self.url_input.text().strip()
        resolution = self.resolution_input.text().strip()
        output_path = self.output_input.text().strip()
        correction_model = self.model_combo.currentText()

        # --- Input Validation ---
        errors = []
        if not playlist_url:
            errors.append("Playlist URL is required.")
        elif not playlist_url.startswith(("http://", "https://")):
            errors.append("Playlist URL seems invalid.")
        if not resolution:
            errors.append("Max Resolution is required.")
        elif not resolution.isdigit() and resolution.lower() != "best":
            errors.append("Resolution must be a number (e.g., 720) or 'best'.")
        if not output_path:
            errors.append("Output Path is required.")
        elif not os.path.isdir(output_path):
            errors.append(f"Output Path is not a valid directory: {output_path}")
        if not os.getenv("OPENROUTER_API_KEY"):
            errors.append("OPENROUTER_API_KEY not found in environment. Please set it.")
        if not workflow_logic.check_ffmpeg():
            errors.append("FFmpeg not found. Cannot proceed without it.")

        if errors:
            QMessageBox.warning(self, "Input Error", "\n".join(errors))
            logging.warning(f"Input validation failed: {errors}")
            return

        logging.info("Input validation passed.")
        self.video_table.setRowCount(0)
        self.error_text.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting...")
        self.status_label.setText("Starting processing...")

        # Create and start the thread
        self.thread = ProcessingThread(
            playlist_url, resolution, output_path, correction_model
        )
        self.thread.status_update.connect(self.update_status)
        self.thread.video_status_update.connect(self.update_video_status)
        self.thread.error_signal.connect(self.handle_error)
        self.thread.progress_update.connect(self.update_progress)
        self.thread.finished_signal.connect(self.on_thread_finished)

        try:
            self.thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            logging.info("Processing thread started successfully.")
        except Exception as e:
            error_msg = f"Failed to start processing thread: {str(e)}"
            QMessageBox.critical(self, "Thread Start Error", error_msg)
            logging.critical(f"{error_msg}")

    def stop_processing(self):
        logging.info("Stop processing button clicked.")
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            logging.info("Processing thread stopped.")

    def update_status(self, status):
        self.status_label.setText(status)
        logging.info(f"Status updated: {status}")

    def update_video_status(self, video_index, status, details):
        while self.video_table.rowCount() <= video_index:
            self.video_table.insertRow(video_index)
        self.video_table.setItem(video_index, 0, QTableWidgetItem(str(video_index + 1)))
        self.video_table.setItem(video_index, 1, QTableWidgetItem(details))
        self.video_table.setItem(video_index, 2, QTableWidgetItem(status))
        self.video_table.scrollToBottom()  # Scroll to the latest entry

    def handle_error(self, video_index, error_message):
        while self.video_table.rowCount() <= video_index:
            self.video_table.insertRow(video_index)
        self.video_table.setItem(video_index, 0, QTableWidgetItem(str(video_index + 1)))
        self.video_table.setItem(video_index, 1, QTableWidgetItem("Error"))
        self.video_table.setItem(video_index, 2, QTableWidgetItem(error_message))
        self.video_table.scrollToBottom()  # Scroll to the latest entry

    def update_progress(self, current_count, total_count):
        self.progress_bar.setValue(int((current_count / total_count) * 100))
        self.progress_bar.setFormat(f"Processing: {current_count}/{total_count}")

    def on_thread_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Processing complete.")
        logging.info("Processing thread finished.")

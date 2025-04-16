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
from PyQt6.QtGui import QAction, QKeySequence
from dotenv import load_dotenv
import yt_dlp

# Import the refactored workflow logic
from src.utils import workflow_logic
from src.utils import model_manager

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

    def __init__(
        self,
        playlist_url,
        resolution,
        output_path,
        correction_model,
        summarization_model,
    ):
        super().__init__()
        self.playlist_url = playlist_url
        self.resolution = resolution
        self.output_path = output_path
        self.correction_model = correction_model
        self.summarization_model = summarization_model
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
                summary_path = os.path.join(
                    video_folder_path, f"{sanitized_title}_summary.txt"
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

                    # --- 4. Diarize Speakers (always) ---
                    if not self._is_running:
                        break
                    self.set_operation(f"Diarizing speakers for {video_title}")
                    self.video_status_update.emit(index, "Diarizing", "")
                    try:
                        diarization_result = workflow_logic.diarize_speakers(
                            wav_audio_path
                        )
                        with open(diarization_path, "w", encoding="utf-8") as f:
                            json.dump(diarization_result, f, indent=2)
                        logging.info(f"Diarization results saved: {diarization_path}")
                    except Exception as e:
                        logging.error(
                            f"Diarization step failed for {video_title}: {str(e)}",
                            exc_info=True,
                        )
                        self.error_signal.emit(index, f"Diarization Failed: {str(e)}")
                        self.video_status_update.emit(
                            index, "Warning", "Diarization Failed"
                        )
                        diarization_result = []
                    finally:
                        self.clear_operation()

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

                    # --- 6. Summarize Transcript ---
                    if not self._is_running:
                        break
                    self.set_operation(f"Summarizing transcript for {video_title}")
                    self.video_status_update.emit(index, "Summarizing", "")
                    try:
                        summary = workflow_logic.summarize_transcript(
                            corrected_transcript, self.summarization_model
                        )
                        with open(summary_path, "w", encoding="utf-8") as f:
                            f.write(summary)
                        logging.info(f"Summary saved: {summary_path}")
                    except Exception as e:
                        raise RuntimeError(f"Summarization failed: {str(e)}")
                    finally:
                        self.clear_operation()

                    # --- Mark as Complete ---
                    self.video_status_update.emit(index, "Complete", summary_path)
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Playlist Advanced Processor")
        self.setGeometry(100, 100, 950, 800)
        logging.info("Initializing MainWindow.")

        self.thread = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Input Area ---
        input_group = QWidget()
        input_layout = QGridLayout(input_group)

        # Playlist URL
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube Playlist URL")
        self.url_input.setToolTip(
            "Paste the full URL of the YouTube playlist you want to process."
        )
        url_label = QLabel("Playlist URL:")
        input_layout.addWidget(url_label, 0, 0)
        input_layout.addWidget(self.url_input, 0, 1, 1, 3)

        # Resolution
        self.resolution_input = QLineEdit("720")
        self.resolution_input.setPlaceholderText("e.g., 1080, 720, best")
        self.resolution_input.setToolTip(
            "Maximum video resolution to download (e.g., 720, 1080). 'best' attempts highest available."
        )
        res_label = QLabel("Max Video Res:")
        input_layout.addWidget(res_label, 1, 0)
        input_layout.addWidget(self.resolution_input, 1, 1)

        # Correction Model
        self.correction_search = QLineEdit()
        self.correction_search.setPlaceholderText("Search Correction Models...")
        self.correction_search.setToolTip("Filter the correction model list by name.")
        input_layout.addWidget(self.correction_search, 2, 0, 1, 2)

        self.correction_model_combo = QComboBox()
        self.correction_model_combo.setToolTip(
            "Select the AI model to use for correcting the transcript."
        )
        self.correction_price_label = QLabel("Estimated Correction Cost: $0.00")
        correction_label = QLabel("Correction Model:")
        input_layout.addWidget(correction_label, 3, 0)
        input_layout.addWidget(self.correction_model_combo, 3, 1)
        input_layout.addWidget(self.correction_price_label, 3, 2, 1, 2)

        # Summarization Model
        self.summarization_search = QLineEdit()
        self.summarization_search.setPlaceholderText("Search Summarization Models...")
        self.summarization_search.setToolTip(
            "Filter the summarization model list by name."
        )
        input_layout.addWidget(self.summarization_search, 4, 0, 1, 2)

        self.summarization_model_combo = QComboBox()
        self.summarization_model_combo.setToolTip(
            "Select the AI model to use for summarizing the transcript."
        )
        self.summarization_price_label = QLabel("Estimated Summarization Cost: $0.00")
        summarization_label = QLabel("Summarization Model:")
        input_layout.addWidget(summarization_label, 5, 0)
        input_layout.addWidget(self.summarization_model_combo, 5, 1)
        input_layout.addWidget(self.summarization_price_label, 5, 2, 1, 2)

        # Refresh Models Button
        self.refresh_models_button = QPushButton("Refresh Models")
        self.refresh_models_button.setToolTip(
            "Fetch the latest available AI models from OpenRouter."
        )
        self.refresh_models_button.clicked.connect(self.populate_llm_dropdowns)
        input_layout.addWidget(self.refresh_models_button, 5, 3)

        # Output Path
        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select Output Directory")
        self.output_input.setToolTip(
            "The folder where processed video subfolders will be saved."
        )
        self.output_button = QPushButton("Browse...")
        self.output_button.setToolTip("Choose the main output directory.")
        self.output_button.clicked.connect(self.browse_output_directory)
        output_label = QLabel("Output Path:")
        input_layout.addWidget(output_label, 6, 0)
        input_layout.addWidget(self.output_input, 6, 1, 1, 2)
        input_layout.addWidget(self.output_button, 6, 3)

        main_layout.addWidget(input_group)

        # --- Control Buttons ---
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Processing")
        self.start_button.setToolTip(
            "Begin processing the playlist with the selected settings (Shortcut: Enter/Return)"
        )
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setShortcut(QKeySequence(Qt.Key.Key_Return))
        self.start_button.setAutoDefault(
            True
        )  # Makes it the default button for Enter key
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.setToolTip("Halt the current processing job (Shortcut: Esc)")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.stop_button.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        button_layout.addWidget(self.stop_button)
        main_layout.addLayout(button_layout)

        # --- Progress & Status ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Waiting to start...")
        self.progress_bar.setToolTip("Overall progress of the playlist processing.")
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel(
            "Ready. Ensure OPENROUTER_API_KEY and HF_TOKEN (optional) are set in .env file or environment."
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setToolTip(
            "Current status of the application or processing stage."
        )
        main_layout.addWidget(self.status_label)

        # --- Video Table ---
        self.video_table = QTableWidget(0, 3)
        self.video_table.setHorizontalHeaderLabels(
            ["#", "Video Title", "Status/Result"]
        )
        self.video_table.setColumnWidth(0, 40)
        self.video_table.setColumnWidth(1, 400)
        self.video_table.horizontalHeader().setStretchLastSection(True)
        self.video_table.setToolTip("Detailed status for each video in the playlist.")
        main_layout.addWidget(self.video_table)

        # --- Error Log ---
        log_label = QLabel("Log / Errors:")
        main_layout.addWidget(log_label)
        self.error_text = QTextEdit()
        self.error_text.setReadOnly(True)
        self.error_text.setMaximumHeight(150)
        self.error_text.setToolTip(
            "Displays processing logs and any errors encountered."
        )
        main_layout.addWidget(self.error_text)

        # --- Set Tab Order ---
        self.setTabOrder(self.url_input, self.resolution_input)
        self.setTabOrder(self.resolution_input, self.correction_search)
        self.setTabOrder(self.correction_search, self.correction_model_combo)
        self.setTabOrder(self.correction_model_combo, self.summarization_search)
        self.setTabOrder(self.summarization_search, self.summarization_model_combo)
        self.setTabOrder(self.summarization_model_combo, self.refresh_models_button)
        self.setTabOrder(self.refresh_models_button, self.output_input)
        self.setTabOrder(self.output_input, self.output_button)
        self.setTabOrder(self.output_button, self.start_button)
        self.setTabOrder(self.start_button, self.stop_button)
        self.setTabOrder(self.stop_button, self.video_table)  # Table can receive focus
        self.setTabOrder(self.video_table, self.error_text)  # Log text area

        # Check for FFmpeg on startup
        QTimer.singleShot(100, self.initial_checks)

        # Populate LLM dropdowns
        self.populate_llm_dropdowns()

        # Connect dropdown changes to price update
        self.correction_model_combo.currentIndexChanged.connect(
            self.update_correction_price
        )
        self.summarization_model_combo.currentIndexChanged.connect(
            self.update_summarization_price
        )

        logging.info("MainWindow UI setup complete.")

    def populate_llm_dropdowns(self):
        config = model_manager.load_config("config/config.yaml")
        llm_models = model_manager.get_available_models(config)
        self.llm_models = llm_models if llm_models else []
        # Use search text to filter models
        correction_search = self.correction_search.text().strip().lower()
        summarization_search = self.summarization_search.text().strip().lower()

        def filter_models(search):
            if not search:
                return self.llm_models
            return [
                m
                for m in self.llm_models
                if search in m["name"].lower()
                or search in m["id"].lower()
                or search in (m.get("description") or "").lower()
            ]

        correction_models = filter_models(correction_search)
        summarization_models = filter_models(summarization_search)
        self.correction_model_combo.clear()
        self.summarization_model_combo.clear()
        for model in correction_models:
            display_name = f"{model['name']} ({model['id']})"
            tooltip = (
                f"Provider: {model.get('provider','')}\n"
                f"Context: {model.get('features',{}).get('max_context','?')}\n"
                f"Input Price: {model.get('input_price_per_million')}\n"
                f"Output Price: {model.get('output_price_per_million')}\n"
                f"Variants: {', '.join(model.get('features',{}).get('variants',[]))}\n"
                f"Tool Calling: {'Yes' if model.get('features',{}).get('tool_calling') else 'No'}\n"
                f"Structured Outputs: {'Yes' if model.get('features',{}).get('structured_outputs') else 'No'}\n"
                f"Description: {model.get('description','')}"
            )
            self.correction_model_combo.addItem(display_name, model)
            self.correction_model_combo.setItemData(
                self.correction_model_combo.count() - 1,
                tooltip,
                Qt.ItemDataRole.ToolTipRole,
            )
        for model in summarization_models:
            display_name = f"{model['name']} ({model['id']})"
            tooltip = (
                f"Provider: {model.get('provider','')}\n"
                f"Context: {model.get('features',{}).get('max_context','?')}\n"
                f"Input Price: {model.get('input_price_per_million')}\n"
                f"Output Price: {model.get('output_price_per_million')}\n"
                f"Variants: {', '.join(model.get('features',{}).get('variants',[]))}\n"
                f"Tool Calling: {'Yes' if model.get('features',{}).get('tool_calling') else 'No'}\n"
                f"Structured Outputs: {'Yes' if model.get('features',{}).get('structured_outputs') else 'No'}\n"
                f"Description: {model.get('description','')}"
            )
            self.summarization_model_combo.addItem(display_name, model)
            self.summarization_model_combo.setItemData(
                self.summarization_model_combo.count() - 1,
                tooltip,
                Qt.ItemDataRole.ToolTipRole,
            )
        self.update_correction_price()
        self.update_summarization_price()
        # Error handling
        if not self.llm_models:
            self.status_label.setText(
                "Error: No LLM models found. Check your network, API key, or credits."
            )
        elif not correction_models or not summarization_models:
            self.status_label.setText(
                "No models match your search. Try a different search term or refresh models."
            )

        # Connect search boxes to update dropdowns live
        self.correction_search.textChanged.connect(self.populate_llm_dropdowns)
        self.summarization_search.textChanged.connect(self.populate_llm_dropdowns)

    def estimate_price(self, model, num_input_tokens, num_output_tokens):
        # Prices are per 1M tokens
        input_price = model.get("input_price_per_million") or 0
        output_price = model.get("output_price_per_million") or 0
        cost = (
            input_price * num_input_tokens + output_price * num_output_tokens
        ) / 1_000_000
        return cost

    def update_correction_price(self):
        model = self.correction_model_combo.currentData()
        if not model:
            self.correction_price_label.setText("Estimated Correction Cost: $0.00")
            return
        # For estimation, assume 10,000 input and 10,000 output tokens
        cost = self.estimate_price(model, 10000, 10000)
        self.correction_price_label.setText(f"Estimated Correction Cost: ${cost:.2f}")

    def update_summarization_price(self):
        model = self.summarization_model_combo.currentData()
        if not model:
            self.summarization_price_label.setText(
                "Estimated Summarization Cost: $0.00"
            )
            return
        # For estimation, assume 2,000 input and 2,000 output tokens
        cost = self.estimate_price(model, 2000, 2000)
        self.summarization_price_label.setText(
            f"Estimated Summarization Cost: ${cost:.2f}"
        )

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
        correction_model = self.correction_model_combo.currentData()
        summarization_model = self.summarization_model_combo.currentData()

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
        if not correction_model:
            errors.append("Please select a correction model.")
        if not summarization_model:
            errors.append("Please select a summarization model.")

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
            playlist_url,
            resolution,
            output_path,
            correction_model["id"],
            summarization_model["id"],
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

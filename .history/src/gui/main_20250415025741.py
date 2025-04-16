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

    # ... (rest of ProcessingThread unchanged, but pass self.summarization_model to workflow_logic as needed)


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

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube Playlist URL")
        input_layout.addWidget(QLabel("Playlist URL:"), 0, 0)
        input_layout.addWidget(self.url_input, 0, 1, 1, 3)

        self.resolution_input = QLineEdit("720")
        self.resolution_input.setPlaceholderText("e.g., 1080, 720, best")
        input_layout.addWidget(QLabel("Max Video Res:"), 1, 0)
        input_layout.addWidget(self.resolution_input, 1, 1)

        # --- Correction Model Dropdown ---
        self.correction_model_combo = QComboBox()
        self.correction_price_label = QLabel("Estimated Correction Cost: $0.00")
        input_layout.addWidget(QLabel("Correction Model:"), 2, 0)
        input_layout.addWidget(self.correction_model_combo, 2, 1)
        input_layout.addWidget(self.correction_price_label, 2, 2, 1, 2)

        # --- Summarization Model Dropdown ---
        self.summarization_model_combo = QComboBox()
        self.summarization_price_label = QLabel("Estimated Summarization Cost: $0.00")
        input_layout.addWidget(QLabel("Summarization Model:"), 3, 0)
        input_layout.addWidget(self.summarization_model_combo, 3, 1)
        input_layout.addWidget(self.summarization_price_label, 3, 2, 1, 2)

        self.output_input = QLineEdit()
        self.output_input.setPlaceholderText("Select Output Directory")
        self.output_button = QPushButton("Browse...")
        self.output_button.clicked.connect(self.browse_output_directory)
        input_layout.addWidget(QLabel("Output Path:"), 4, 0)
        input_layout.addWidget(self.output_input, 4, 1, 1, 2)
        input_layout.addWidget(self.output_button, 4, 3)

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
        llm_models = model_manager.get_available_llm_models_with_pricing(config)
        self.llm_models = llm_models if llm_models else []
        self.correction_model_combo.clear()
        self.summarization_model_combo.clear()
        for model in self.llm_models:
            display_name = f"{model['name']} ({model['id']})"
            self.correction_model_combo.addItem(display_name, model)
            self.summarization_model_combo.addItem(display_name, model)
        self.update_correction_price()
        self.update_summarization_price()

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

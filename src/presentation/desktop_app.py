import sys
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit,
    QPushButton, QListWidget, QAbstractItemView, QHBoxLayout, QCheckBox,
    QRadioButton, QButtonGroup, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QScrollArea, QFrame, QComboBox, QAbstractButton, QSizePolicy,
    QTextEdit, QFileDialog # Added QFileDialog
)
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, QSize, Signal # Added Signal back

import yt_dlp
import requests
import json
import os
import tiktoken # Assuming tiktoken is installed for token estimation

# Worker thread for running backend processes
class WorkerThread(QThread):
    progress_update = Signal(str)
    video_status_update = Signal(str, str) # Signal to update status of a specific video (video_id, status_message)
    error_occurred = Signal(str, str) # New signal for error messages (video_id, error_message)
    process_finished = Signal(int) # Signal to indicate process finished with exit code

    def __init__(self, command, video_ids=None):
        super().__init__()
        self.command = command
        self.video_ids = video_ids # List of video IDs being processed by this thread

    def run(self):
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Read stdout and stderr in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Attempt to parse structured progress updates from backend scripts
                    # EXPECTED FORMAT: "VIDEO_STATUS:<video_id>:<status_message>"
                    if output.startswith("VIDEO_STATUS:"):
                        parts = output.strip().split(":", 2)
                        if len(parts) == 3:
                            video_id = parts[1]
                            status_message = parts[2]
                            self.video_status_update.emit(video_id, status_message)
                        else:
                            self.progress_update.emit(output.strip()) # Emit as general progress if parsing fails
                    # EXPECTED FORMAT: "VIDEO_ERROR:<video_id>:<error_message>"
                    elif output.startswith("VIDEO_ERROR:"):
                         parts = output.strip().split(":", 2)
                         if len(parts) == 3:
                             video_id = parts[1]
                             error_message = parts[2]
                             self.error_occurred.emit(video_id, error_message)
                         else:
                             self.progress_update.emit(output.strip()) # Emit as general progress if parsing fails
                    else:
                        self.progress_update.emit(output.strip()) # Emit as general progress

            # Read remaining stderr
            stderr_output = process.stderr.read()
            if stderr_output:
                 # If stderr is not structured, emit as a general error
                 self.progress_update.emit(f"Error: {stderr_output.strip()}")
                 # TODO: Implement more sophisticated error parsing and reporting for unstructured stderr

            exit_code = process.wait()
            self.process_finished.emit(exit_code)

        except Exception as e:
            self.progress_update.emit(f"Error executing command: {e}")
            self.process_finished.emit(1) # Indicate error with non-zero exit code


# Basic structure for the main application window
class MainWindow(QMainWindow):
    # Define signals here
    # video_info_fetched = Signal(list) # Example signal
    # progress_update = Signal(str) # Signal for status updates

    def __init__(self):
        super().__init__()

        self.setWindowTitle("YouTube Playlist Processor")
        self.setGeometry(100, 100, 1200, 800) # Increased window size

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Add widgets for the UI based on the plan
        # Input for playlist URL
        url_layout = QHBoxLayout()
        main_layout.addLayout(url_layout)
        url_label = QLabel("Playlist URL:")
        url_layout.addWidget(url_label)
        self.url_input = QLineEdit()
        url_layout.addWidget(self.url_input)
        self.fetch_button = QPushButton("Fetch Videos")
        url_layout.addWidget(self.fetch_button)

        # Input/Output Path Configuration (Placeholder)
        path_groupbox = QGroupBox("Path Configuration")
        path_layout = QVBoxLayout()
        path_groupbox.setLayout(path_layout)
        main_layout.addWidget(path_groupbox)

        output_path_layout = QHBoxLayout()
        path_layout.addLayout(output_path_layout)
        output_path_label = QLabel("Output Directory:")
        output_path_layout.addWidget(output_path_label)
        self.output_path_input = QLineEdit()
        self.output_path_input.setReadOnly(True) # Make it read-only, use button to select
        output_path_layout.addWidget(self.output_path_input)
        self.browse_output_button = QPushButton("Browse")
        output_path_layout.addWidget(self.browse_output_button)

        # TODO: Add input path configuration if needed by backend scripts

        # Main content area: Video table and sidebar
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout)

        # Area to display video list (using QTableWidget for better structure)
        self.video_table = QTableWidget()
        # Added columns for Duration and Views
        self.video_table.setColumnCount(6)
        self.video_table.setHorizontalHeaderLabels(["Select", "Thumbnail", "Title", "Duration", "Views", "Status"])
        self.video_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # Select column
        self.video_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents) # Thumbnail column
        self.video_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch) # Title column
        self.video_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) # Duration column
        self.video_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents) # Views column
        self.video_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents) # Status column
        self.video_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.video_table.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        content_layout.addWidget(self.video_table, 3) # Give video table more space

        # Sidebar for model selection and cost estimation
        sidebar_scroll_area = QScrollArea()
        sidebar_scroll_area.setWidgetResizable(True)
        sidebar_scroll_area.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        sidebar_scroll_area.setMinimumWidth(250) # Set a minimum width for the sidebar
        content_layout.addWidget(sidebar_scroll_area, 1) # Give sidebar less space

        sidebar_frame = QFrame()
        sidebar_layout = QVBoxLayout()
        sidebar_frame.setLayout(sidebar_layout)
        sidebar_scroll_area.setWidget(sidebar_frame)

        # Model selection for summarization
        model_groupbox = QGroupBox("Summarization Model")
        model_groupbox_layout = QVBoxLayout()
        model_groupbox.setLayout(model_groupbox_layout)
        sidebar_layout.addWidget(model_groupbox)

        model_layout = QHBoxLayout()
        model_groupbox_layout.addLayout(model_layout)
        model_label = QLabel("Model:")
        model_layout.addWidget(model_label)
        self.model_dropdown = QComboBox()
        model_layout.addWidget(self.model_dropdown)

        self.fetch_models_button = QPushButton("Fetch Models") # Button to trigger fetching models
        model_groupbox_layout.addWidget(self.fetch_models_button)

        # Estimated cost display
        cost_groupbox = QGroupBox("Estimated Cost")
        cost_groupbox_layout = QVBoxLayout()
        cost_groupbox.setLayout(cost_groupbox_layout)
        sidebar_layout.addWidget(cost_groupbox)

        self.cost_label = QLabel("Estimated Cost: N/A")
        cost_groupbox_layout.addWidget(self.cost_label)

        # Add stretch to push everything to the top of the sidebar
        sidebar_layout.addStretch(1)


        # Buttons for actions
        button_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        self.download_button = QPushButton("Download Selected")
        button_layout.addWidget(self.download_button)
        self.summarize_button = QPushButton("Summarize Transcripts")
        button_layout.addWidget(self.summarize_button)
        self.run_pipeline_button = QPushButton("Run Full Pipeline")
        button_layout.addWidget(self.run_pipeline_button)

        # Status area
        self.status_label = QLabel("Status: Idle")
        main_layout.addWidget(self.status_label)

        # Error log area
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setWindowTitle("Error Log") # Set window title for potential undocking
        self.error_log.hide() # Hide initially, show when errors occur
        main_layout.addWidget(self.error_log)


        # Connect signals and slots
        self.fetch_button.clicked.connect(self.fetch_videos)
        self.download_button.clicked.connect(self.download_videos) # Connected
        self.summarize_button.clicked.connect(self.summarize_transcripts) # Connected
        self.run_pipeline_button.clicked.connect(self.run_full_pipeline) # Connected
        self.fetch_models_button.clicked.connect(self.fetch_openrouter_models)
        self.model_dropdown.currentIndexChanged.connect(self.update_estimated_cost) # Connect model selection to cost update
        self.browse_output_button.clicked.connect(self.browse_output_directory) # Connect browse button

        # Initialize worker thread attribute
        self.worker_thread = None
        self.openrouter_models_data = {} # Store fetched model data including pricing
        self.video_data = {} # Store video data by ID for easy access

    def browse_output_directory(self):
        """Opens a dialog to select the output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_path_input.setText(directory)

    def fetch_videos(self):
        playlist_url = self.url_input.text()
        if not playlist_url:
            self.status_label.setText("Status: Please enter a playlist URL.")
            return

        self.status_label.setText("Status: Fetching video information...")
        self.video_table.setRowCount(0) # Clear previous results
        self.video_data = {} # Clear previous video data
        self.error_log.clear() # Clear error log
        self.error_log.hide() # Hide error log

        try:
            ydl_opts = {
                "quiet": True,
                "extract_flat": "in_playlist",
                "skip_download": True,
                "forceurl": True,
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                },
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                video_list = info.get("entries", [])

            if not video_list:
                self.status_label.setText("Status: No videos found in the playlist.")
                return

            self.video_table.setRowCount(len(video_list))
            for row, video in enumerate(video_list):
                video_id = video.get("id")
                if not video_id:
                    continue # Skip if no video ID

                self.video_data[video_id] = video # Store video data by ID

                # Add checkbox
                checkbox_item = QTableWidgetItem()
                checkbox_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                checkbox_item.setCheckState(Qt.CheckState.Unchecked)
                self.video_table.setItem(row, 0, checkbox_item)

                # Add thumbnail
                thumbnail_url = video.get("thumbnail")
                if thumbnail_url:
                    try:
                        response = requests.get(thumbnail_url, stream=True)
                        response.raise_for_status()
                        image = QImage.fromData(response.content)
                        if not image.isNull():
                            pixmap = QPixmap.fromImage(image)
                            # Scale thumbnail to a reasonable size (e.g., 80x60)
                            scaled_pixmap = pixmap.scaled(80, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                            thumbnail_label = QLabel()
                            thumbnail_label.setPixmap(scaled_pixmap)
                            self.video_table.setCellWidget(row, 1, thumbnail_label)
                            self.video_table.setRowHeight(row, 60) # Adjust row height to fit thumbnail
                        else:
                            thumbnail_item = QTableWidgetItem("Invalid Thumbnail")
                            self.video_table.setItem(row, 1, thumbnail_item)
                    except Exception as thumbnail_e:
                        thumbnail_item = QTableWidgetItem(f"Error: {thumbnail_e}")
                        self.video_table.setItem(row, 1, thumbnail_item)
                else:
                    thumbnail_item = QTableWidgetItem("No Thumbnail")
                    self.video_table.setItem(row, 1, thumbnail_item)


                # Add title and store video ID in UserRole
                title_item = QTableWidgetItem(video.get("title", "Unknown Title"))
                title_item.setData(Qt.ItemDataRole.UserRole, video_id) # Store video ID
                self.video_table.setItem(row, 2, title_item)

                # Add Duration
                duration = video.get("duration")
                duration_item = QTableWidgetItem(f"{duration}s" if duration is not None else "N/A")
                self.video_table.setItem(row, 3, duration_item)

                # Add Views
                view_count = video.get("view_count")
                view_count_item = QTableWidgetItem(f"{view_count:,}" if view_count is not None else "N/A")
                self.video_table.setItem(row, 4, view_count_item)


                # Add status
                status_item = QTableWidgetItem("Pending")
                self.video_table.setItem(row, 5, status_item) # Status is now column 5

            self.status_label.setText(f"Status: Found {len(video_list)} videos.")

        except Exception as e:
            self.status_label.setText(f"Status: Error fetching videos - {e}")
            self.error_log.append(f"Error fetching videos: {e}")
            self.error_log.show()


    def fetch_openrouter_models(self):
        self.status_label.setText("Status: Fetching OpenRouter models...")
        self.model_dropdown.clear() # Clear previous models
        self.openrouter_models_data = {} # Clear previous model data
        self.error_log.clear() # Clear error log
        self.error_log.hide() # Hide error log


        try:
            # Assuming the API key is stored in an environment variable
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                self.status_label.setText("Status: OPENROUTER_API_KEY not set in environment variables.")
                self.error_log.append("Error: OPENROUTER_API_KEY not set in environment variables.")
                self.error_log.show()
                return

            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
            response.raise_for_status()
            models_data = response.json()

            models_list = models_data.get("data", [])

            if not models_list:
                self.status_label.setText("Status: No OpenRouter models found.")
                return

            for model in models_list:
                model_id = model.get("id")
                model_name = model.get("name", model_id)
                pricing = model.get("pricing", {})
                # Store relevant model data including pricing
                self.openrouter_models_data[model_id] = {
                    "name": model_name,
                    "pricing": pricing
                }
                self.model_dropdown.addItem(model_name, model_id) # Store model_id as UserData

            self.status_label.setText(f"Status: Loaded {len(models_list)} OpenRouter models.")
            self.update_estimated_cost() # Update cost after loading models

        except Exception as e:
            self.status_label.setText(f"Status: Error fetching OpenRouter models - {e}")
            self.error_log.append(f"Error fetching OpenRouter models: {e}")
            self.error_log.show()
            self.openrouter_models_data = {} # Clear data on error
            self.update_estimated_cost() # Update cost to N/A


    def get_selected_videos(self):
        selected_videos = []
        for row in range(self.video_table.rowCount()):
            checkbox_item = self.video_table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                title_item = self.video_table.item(row, 2)
                video_id = title_item.data(Qt.ItemDataRole.UserRole) # Get video ID from UserRole
                if video_id and video_id in self.video_data:
                     selected_videos.append(self.video_data[video_id]) # Append full video data
        return selected_videos

    def estimate_transcript_tokens(self, video_id):
        """
        Estimates the token count of a video transcript based on video duration.
        Assumes an average speech rate to estimate character count, then converts to tokens.
        """
        video_info = self.video_data.get(video_id)
        if not video_info or video_info.get("duration") is None:
            return 0 # Cannot estimate if no video info or duration

        duration_seconds = video_info["duration"]
        # Estimate speech rate: average words per minute * characters per word
        # Assuming 150 words per minute and 5 characters per word + space = 6 characters per word
        estimated_characters_per_second = (150 / 60) * 6
        estimated_character_count = duration_seconds * estimated_characters_per_second

        # Estimate tokens using a simple character-to-token ratio (e.g., 4 characters per token for English)
        # A more accurate method would use a tokenizer like tiktoken
        estimated_tokens = estimated_character_count / 4

        return int(estimated_tokens)


    def calculate_estimated_cost(self, model_id, prompt_tokens, completion_tokens):
        """Calculates the estimated cost based on model pricing and token counts."""
        model_data = self.openrouter_models_data.get(model_id)
        if not model_data or "pricing" not in model_data:
            return None # Pricing information not available

        pricing = model_data["pricing"]
        prompt_cost_per_million = pricing.get("prompt")
        completion_cost_per_million = pricing.get("completion")

        if prompt_cost_per_million is None or completion_cost_per_million is None:
            return None # Pricing information incomplete

        estimated_cost = (prompt_tokens / 1_000_000) * prompt_cost_per_million + \
                         (completion_tokens / 1_000_000) * completion_cost_per_million

        return estimated_cost

    def update_estimated_cost(self):
        """Updates the estimated cost label based on selected model and selected videos."""
        selected_model_id = self.model_dropdown.currentData(Qt.ItemDataRole.UserRole)
        selected_videos = self.get_selected_videos()

        if not selected_model_id or not selected_videos:
            self.cost_label.setText("Estimated Cost: N/A")
            return

        total_estimated_prompt_tokens = 0
        for video in selected_videos:
            # Estimate tokens for each selected video's transcript
            total_estimated_prompt_tokens += self.estimate_transcript_tokens(video.get("id"))

        # Assuming summarization might generate roughly 10% of the transcript length in tokens
        estimated_completion_tokens = total_estimated_prompt_tokens * 0.1

        estimated_cost = self.calculate_estimated_cost(
            selected_model_id,
            total_estimated_prompt_tokens,
            estimated_completion_tokens
        )

        if estimated_cost is not None:
            self.cost_label.setText(f"Estimated Cost: ${estimated_cost:.6f}")
        else:
            self.cost_label.setText("Estimated Cost: Pricing N/A")


    def download_videos(self):
        """Initiates the download of selected videos using a backend script."""
        selected_videos = self.get_selected_videos()
        if not selected_videos:
            self.status_label.setText("Status: No videos selected for download.")
            return

        playlist_url = self.url_input.text()
        if not playlist_url:
             self.status_label.setText("Status: Cannot download. Playlist URL is missing.")
             return

        output_path = self.output_path_input.text()
        if not output_path:
            self.status_label.setText("Status: Please select an output directory.")
            return


        # Extract video IDs from selected videos
        video_ids = [video.get("id") for video in selected_videos if video.get("id")]
        if not video_ids:
             self.status_label.setText("Status: No valid video IDs found for selected videos.")
             return

        # Construct the command to run the download script
        # Assuming the script takes playlist URL, video IDs, and output path as arguments
        command = [
            sys.executable, # Use the same Python interpreter
            "src/presentation/download_handler.py", # Updated path
            "--playlist_url", playlist_url,
            "--video_ids", ",".join(video_ids), # Pass video IDs as a comma-separated string
            "--output_path", output_path # Pass output path
        ]

        self.status_label.setText(f"Status: Starting download for {len(selected_videos)} videos...")
        self.error_log.clear() # Clear error log for new process
        self.error_log.hide() # Hide error log initially


        # Run the download script in a worker thread
        self.worker_thread = WorkerThread(command, video_ids=video_ids) # Pass video_ids to thread
        self.worker_thread.progress_update.connect(self.update_status)
        self.worker_thread.video_status_update.connect(self.update_video_status) # Connect video status signal
        self.worker_thread.error_occurred.connect(self.handle_error) # Connect error signal
        self.worker_thread.process_finished.connect(self.download_finished)
        self.worker_thread.start()

    def download_finished(self, exit_code):
        """Handles the completion of the download process."""
        if exit_code == 0:
            self.status_label.setText("Status: Download process finished successfully.")
            # TODO: Update status of downloaded videos in the table if not already done by video_status_update
        else:
            self.status_label.setText(f"Status: Download process failed with exit code {exit_code}.")
            # TODO: Update status of videos that failed to download
        self.worker_thread = None # Clear the worker thread


    def summarize_transcripts(self):
        """Initiates the summarization of selected video transcripts."""
        selected_videos = self.get_selected_videos()
        if not selected_videos:
            self.status_label.setText("Status: No videos selected for summarization.")
            return

        selected_model_name = self.model_dropdown.currentText()
        selected_model_id = self.model_dropdown.currentData(Qt.ItemDataRole.UserRole)

        if not selected_model_id: # Check if a valid model is selected
             self.status_label.setText("Status: Please select a summarization model.")
             return

        output_path = self.output_path_input.text()
        if not output_path:
            self.status_label.setText("Status: Please select an output directory.")
            return


        # Extract video IDs from selected videos
        video_ids = [video.get("id") for video in selected_videos if video.get("id")]
        if not video_ids:
             self.status_label.setText("Status: No valid video IDs found for selected videos.")
             return

        # Construct the command to run the summarization script
        # Assuming the script takes video IDs, model name, and output path as arguments
        command = [
            sys.executable, # Use the same Python interpreter
            "src/presentation/summarize_handler.py", # Updated path
            "--video_ids", ",".join(video_ids), # Pass video IDs
            "--model_name", selected_model_id, # Pass selected model ID
            "--output_path", output_path # Pass output path
            # TODO: Add arguments for input paths if needed (e.g., path to raw transcripts)
        ]

        self.status_label.setText(f"Status: Starting summarization for {len(selected_videos)} videos using {selected_model_name}...")
        self.error_log.clear() # Clear error log for new process
        self.error_log.hide() # Hide error log initially


        # Run the summarization script in a worker thread
        self.worker_thread = WorkerThread(command, video_ids=video_ids) # Pass video_ids to thread
        self.worker_thread.progress_update.connect(self.update_status)
        self.worker_thread.video_status_update.connect(self.update_video_status) # Connect video status signal
        self.worker_thread.error_occurred.connect(self.handle_error) # Connect error signal
        self.worker_thread.process_finished.connect(self.summarize_finished)
        self.worker_thread.start()

    def summarize_finished(self, exit_code):
        """Handles the completion of the summarization process."""
        if exit_code == 0:
            self.status_label.setText("Status: Summarization process finished successfully.")
            # TODO: Update status of summarized videos in the table if not already done by video_status_update
        else:
            self.status_label.setText(f"Status: Summarization process failed with exit code {exit_code}.")
            # TODO: Update status of videos that failed to summarize
        self.worker_thread = None # Clear the worker thread


    def run_full_pipeline(self):
        """Initiates the full pipeline for selected videos."""
        selected_videos = self.get_selected_videos()
        if not selected_videos:
            self.status_label.setText("Status: No videos selected to run the full pipeline.")
            return

        selected_model_name = self.model_dropdown.currentText()
        selected_model_id = self.model_dropdown.currentData(Qt.ItemDataRole.UserRole)

        if not selected_model_id: # Check if a valid model is selected
             self.status_label.setText("Status: Please select a summarization model for the full pipeline.")
             return

        output_path = self.output_path_input.text()
        if not output_path:
            self.status_label.setText("Status: Please select an output directory.")
            return


        # Extract video IDs from selected videos
        video_ids = [video.get("id") for video in selected_videos if video.get("id")]
        if not video_ids:
             self.status_label.setText("Status: No valid video IDs found for selected videos.")
             return

        # Construct the command to run the full pipeline script
        # Assuming the script takes playlist URL, video IDs, model name, and output path as arguments
        playlist_url = self.url_input.text()
        if not playlist_url:
             self.status_label.setText("Status: Cannot run full pipeline. Playlist URL is missing.")
             return

        command = [
            sys.executable, # Use the same Python interpreter
            "src/run_full_pipeline.py", # Assuming this is the full pipeline script
            "--playlist_url", playlist_url,
            "--video_ids", ",".join(video_ids), # Pass video IDs
            "--model_name", selected_model_id, # Pass selected model ID
            "--output_path", output_path # Pass output path
            # TODO: Add arguments for input paths if needed
        ]

        self.status_label.setText(f"Status: Starting full pipeline for {len(selected_videos)} videos using {selected_model_name}...")
        self.error_log.clear() # Clear error log for new process
        self.error_log.hide() # Hide error log initially


        # Run the full pipeline script in a worker thread
        self.worker_thread = WorkerThread(command, video_ids=video_ids) # Pass video_ids to thread
        self.worker_thread.progress_update.connect(self.update_status)
        self.worker_thread.video_status_update.connect(self.update_video_status) # Connect video status signal
        self.worker_thread.error_occurred.connect(self.handle_error) # Connect error signal
        self.worker_thread.process_finished.connect(self.full_pipeline_finished)
        self.worker_thread.start()

    def full_pipeline_finished(self, exit_code):
        """Handles the completion of the full pipeline process."""
        if exit_code == 0:
            self.status_label.setText("Status: Full pipeline process finished successfully.")
            # TODO: Update status of processed videos in the table if not already done by video_status_update
        else:
            self.status_label.setText(f"Status: Full pipeline process failed with exit code {exit_code}.")
            # TODO: Update status of videos that failed in the pipeline
        self.worker_thread = None # Clear the worker thread


    def update_status(self, message):
        """Updates the status label with messages from the worker thread."""
        self.status_label.setText(f"Status: {message}")

    def update_video_status(self, video_id, status_message):
        """Updates the status column for a specific video in the table."""
        for row in range(self.video_table.rowCount()):
            title_item = self.video_table.item(row, 2)
            if title_item and title_item.data(Qt.ItemDataRole.UserRole) == video_id:
                status_item = self.video_table.item(row, 5) # Status is now column 5
                if status_item:
                    status_item.setText(status_message)
                    # Optional: Change text color for error status
                    if "Error" in status_message or "Failed" in status_message:
                        status_item.setForeground(QColor(Qt.GlobalColor.red))
                    else:
                        status_item.setForeground(QColor(Qt.GlobalColor.black))
                break # Found the row, exit loop

    def handle_error(self, video_id, error_message):
        """Handles error messages from the worker thread and displays them in the error log."""
        error_text = f"Error processing video {video_id}: {error_message}"
        self.error_log.append(error_text)
        self.error_log.show() # Show the error log when an error occurs
        # Optionally update the status of the specific video in the table to indicate error
        self.update_video_status(video_id, f"Error: {error_message[:50]}...") # Display truncated error in table

    def cleanup_files(self):
        """
        Placeholder method for cleaning up intermediate files.
        The actual file deletion logic should be in the backend scripts.
        """
        self.status_label.setText("Status: Cleanup button clicked (placeholder).")
        # TODO: If a separate cleanup script is created, call it here in a worker thread.
        # Otherwise, ensure backend scripts handle file deletion after transcription.


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
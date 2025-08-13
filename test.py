import tkinter as tk
import traceback
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from upload_video import VideoUploader
import numpy as np
from PIL import Image, ImageTk
import time
import threading
import os
import subprocess
import json
import tempfile
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd

class GenderAgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.init_state()
        self.create_ui()
        self.create_status_bar()
        self.processing_interval = 0.
        
        # Verify models before loading
        status, message = self.verify_models()
        if not status:
            messagebox.showwarning("Model Files Missing", message)
        
        self.load_models()
        self.history = []
        self.load_history_from_file()
        self.update_history_view()

    def setup_window(self):
        """Configure window with modern dark theme"""
        self.root.title("VisionAI Gender & Age Analyzer Pro")
        self._title = "VisionAI Gender & Age Analyzer Pro"
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg='#1e1e2d')  # Dark blue-gray background
        
        # Custom modern dark theme style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color definitions
        dark_bg = '#1e1e2d'       # Main background
        darker_bg = '#161622'      # Darker elements
        accent = '#3a3a5d'         # Accent color
        highlight = '#4d4d7a'      # Highlight color
        text_color = '#e0e0e0'     # Light text
        success_color = '#2ecc71'  # Green
        warning_color = '#f39c12'  # Orange
        danger_color = '#e74c3c'   # Red
        
        # Configure styles
        self.style.configure('.', 
                           background=dark_bg,
                           foreground=text_color,
                           font=('Segoe UI', 10))
        
        self.style.configure('TFrame', background=dark_bg)
        self.style.configure('TLabel', 
                           background=dark_bg,
                           foreground=text_color)
        
        self.style.configure('TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=8,
                           background=accent,
                           foreground=text_color,
                           borderwidth=0,
                           relief='flat',
                           bordercolor=accent)
        
        self.style.map('TButton',
                      background=[('active', highlight)],
                      foreground=[('active', text_color)],
                      relief=[('active', 'flat')])
        
        self.style.configure('Title.TLabel',
                           font=('Segoe UI', 24, 'bold'),
                           foreground='#ffffff')
        
        self.style.configure('Subtitle.TLabel',
                           font=('Segoe UI', 12),
                           foreground='#bdc3c7')
        
        self.style.configure('Card.TFrame',
                           background=darker_bg,
                           relief=tk.FLAT,
                           borderwidth=0,
                           padding=10)
        
        self.style.configure('Stat.TLabel',
                           font=('Segoe UI', 16, 'bold'),
                           foreground=text_color)
        
        self.style.configure('TNotebook',
                           background=dark_bg,
                           borderwidth=0)
        
        self.style.configure('TNotebook.Tab',
                           font=('Segoe UI', 10, 'bold'),
                           padding=[15, 5],
                           background=darker_bg,
                           foreground=text_color,
                           borderwidth=0)
        
        self.style.map('TNotebook.Tab',
                      background=[('selected', accent)],
                      foreground=[('selected', text_color)],
                      relief=[('selected', 'flat')])
        
        # Configure scrollbars
        self.style.configure('Vertical.TScrollbar',
                           background=darker_bg,
                           troughcolor=dark_bg,
                           bordercolor=dark_bg,
                           arrowcolor=text_color,
                           gripcount=0)
        
        # Configure treeview
        self.style.configure('Treeview',
                           background=darker_bg,
                           foreground=text_color,
                           fieldbackground=darker_bg,
                           borderwidth=0,
                           rowheight=25)
        
        self.style.map('Treeview',
                      background=[('selected', accent)],
                      foreground=[('selected', text_color)])
        
        # Configure entry widgets
        self.style.configure('TEntry',
                           fieldbackground=darker_bg,
                           foreground=text_color,
                           insertcolor=text_color,
                           bordercolor=accent,
                           lightcolor=accent,
                           darkcolor=accent,
                           padding=5)
        
        # Configure progress bar
        self.style.configure('Horizontal.TProgressbar',
                           background=success_color,
                           troughcolor=darker_bg,
                           bordercolor=darker_bg,
                           lightcolor=success_color,
                           darkcolor=success_color)
        
        # Configure radiobuttons and checkbuttons
        self.style.configure('TRadiobutton',
                           background=dark_bg,
                           foreground=text_color)
        
        self.style.configure('TCheckbutton',
                           background=dark_bg,
                           foreground=text_color)

    def init_state(self):
        """Initialize all state variables"""
        self.capture = None
        self.is_capturing = False
        self.current_frame = None
        self.last_processed_time = 0
        self.processing_interval = 0.3
        self.face_net = None
        self.gender_net = None
        self.age_net = None
        self.gender_model = None
        self.age_model = None
        self.face_detection_enabled = True
        self.gender_detection_enabled = True
        self.age_detection_enabled = True
        self.camera_index = 0
        self.male_count = 0
        self.female_count = 0
        self.age_distribution = {}
        self.total_frames = 0
        self.frame_times = []
        self.last_fps_update = time.time()
        self.video_file_path = None
        self.upload_progress = 0
        self.is_processing_file = False
        self.after_id = None
        self.detection_history = []
        self.session_start_time = None
        self.recording = False
        self.output_video = None
        self.face_detection_model = "haarcascade"
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def create_ui(self):
        """Create all UI components with modern layout"""
        # Main container with gradient background
        self.main_frame = ttk.Frame(self.root, style='Card.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with logo and buttons
        self.header_frame = ttk.Frame(self.main_frame, style='Card.TFrame')
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Logo and title section
        title_frame = ttk.Frame(self.header_frame)
        title_frame.pack(side=tk.LEFT, padx=20)
        
        # App title with logo (using emoji as placeholder)
        self.logo_label = ttk.Label(
            title_frame, 
            text="👁️", 
            font=('Segoe UI Emoji', 24),
            style='TLabel'
        )
        self.logo_label.pack(side=tk.LEFT)
        
        self.title_frame = ttk.Frame(title_frame)
        self.title_frame.pack(side=tk.LEFT, padx=10)
        
        self.title_label = ttk.Label(
            self.title_frame, 
            text="VisionAI", 
            style='Title.TLabel'
        )
        self.title_label.pack(anchor=tk.W)
        
        self.subtitle_label = ttk.Label(
            self.title_frame, 
            text="Gender & Age Analyzer Pro", 
            style='Subtitle.TLabel'
        )
        self.subtitle_label.pack(anchor=tk.W)
        
        # Header buttons with modern icons
        btn_frame = ttk.Frame(self.header_frame)
        btn_frame.pack(side=tk.RIGHT, padx=20)
        
        button_style = 'Accent.TButton'
        self.style.configure(button_style,
                           background='#3a3a5d',
                           foreground='#ffffff',
                           padding=10,
                           font=('Segoe UI', 10, 'bold'))
        
        self.settings_button = ttk.Button(
            btn_frame,
            text="⚙️ Settings",
            command=self.open_settings,
            style=button_style
        )
        self.settings_button.pack(side=tk.LEFT, padx=5, ipadx=5)
        
        self.help_button = ttk.Button(
            btn_frame,
            text="❓ Help",
            command=self.show_help,
            style=button_style
        )
        self.help_button.pack(side=tk.LEFT, padx=5, ipadx=5)
        
        self.export_button = ttk.Button(
            btn_frame,
            text="📊 Export Data",
            command=self.export_data,
            style=button_style
        )
        self.export_button.pack(side=tk.LEFT, padx=5, ipadx=5)
        
        # Main content area with notebook (tabs)
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for tabs with modern styling
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Real-time Analysis Tab
        self.realtime_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.realtime_tab, text="  Real-time Analysis  ")
        self.create_realtime_tab()
        
        # Historical Data Tab
        self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.history_tab, text="  Historical Data  ")
        self.create_history_tab()
        
        # Status bar
        self.status_frame = ttk.Frame(self.main_frame, style='Card.TFrame')
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(
            self.status_frame,
            text="Status: Ready",
            foreground="#bdc3c7",
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.version_label = ttk.Label(
            self.status_frame,
            text="v2.2.0 | © 2023 VisionAI",
            foreground="#bdc3c7",
            anchor=tk.E
        )
        self.version_label.pack(side=tk.RIGHT, padx=10)

    def create_realtime_tab(self):
        """Create content for real-time analysis tab with modern layout"""
        # Main container for real-time tab
        realtime_container = ttk.Frame(self.realtime_tab)
        realtime_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - controls and stats (30% width)
        left_panel = ttk.Frame(realtime_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Control Panel with modern card design
        control_frame = ttk.LabelFrame(
            left_panel, 
            text="CAPTURE CONTROLS", 
            padding=15,
            style='Card.TFrame'
        )
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Styled buttons with icons and modern look
        button_style = 'Accent.TButton'
        
        self.start_button = ttk.Button(
            control_frame,
            text="🎥 Start Camera",
            command=lambda: self.launch_script("start_camera.py"),
            style=button_style
        )
        self.start_button.pack(fill=tk.X, pady=5, ipady=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="⏹ Stop Camera",
            command=lambda: self.launch_script("stop_camera.py"),
            state=tk.NORMAL,
            style=button_style
        )
        self.stop_button.pack(fill=tk.X, pady=5, ipady=5)
        
        self.upload_button = ttk.Button(
            control_frame,
            text="📁 Upload Video",
            command=lambda: self.launch_script("upload_video.py"),
            style=button_style
        )
        self.upload_button.pack(fill=tk.X, pady=5, ipady=5)
        
        # Recording button with toggle state
        self.record_button = ttk.Button(
            control_frame,
            text="🔴 Start Recording",
            command=self.toggle_recording,
            style='Danger.TButton'
        )
        self.record_button.pack(fill=tk.X, pady=5, ipady=5)
        self.style.configure('Danger.TButton', background='#e74c3c')
        
        # Stats Panel with modern cards
        stats_frame = ttk.LabelFrame(
            left_panel, 
            text="DETECTION STATISTICS", 
            padding=15,
            style='Card.TFrame'
        )
        stats_frame.pack(fill=tk.X)
        
        # Create stat cards with modern design
        stats = [
            ("👨 Male", "male", "#3498db"),
            ("👩 Female", "female", "#e74c3c"),
            ("⏱ FPS", "fps", "#2ecc71"),
            ("🎯 Accuracy", "accuracy", "#9b59b6"),
            ("⏳ Duration", "duration", "#f39c12")
        ]
        
        for stat in stats:
            self.create_modern_stat_card(stats_frame, stat[0], stat[1], stat[2])
        
        # Right panel - display and logs (70% width)
        right_panel = ttk.Frame(realtime_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Display frame with modern styling
        display_frame = ttk.LabelFrame(
            right_panel, 
            text="LIVE FEED", 
            padding=10,
            style='Card.TFrame'
        )
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for video feed with aspect ratio preservation
        self.canvas = tk.Canvas(
            display_frame, 
            bg="#161622",  # Darker background for video area
            bd=0,
            highlightthickness=0,
            relief='flat'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Overlay for video information
        self.overlay_frame = ttk.Frame(self.canvas, style='Card.TFrame')
        self.canvas.create_window(10, 10, anchor=tk.NW, window=self.overlay_frame)
        
        self.recording_indicator = ttk.Label(
            self.overlay_frame,
            text="",
            foreground="#e74c3c",
            font=('Segoe UI', 10, 'bold'),
            style='TLabel'
        )
        self.recording_indicator.pack()
        
        # Progress bar with modern styling
        self.progress_frame = ttk.Frame(right_panel, style='Card.TFrame')
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            orient=tk.HORIZONTAL,
            length=500,
            mode='determinate',
            style='Horizontal.TProgressbar'
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_label = ttk.Label(
            self.progress_frame,
            text="Ready",
            foreground="#bdc3c7"
        )
        self.progress_label.pack(pady=(0, 5))
        
        self.hide_progress()
        
        # Log console with modern dark theme
        log_frame = ttk.LabelFrame(
            right_panel, 
            text="DETECTION LOG", 
            padding=10,
            style='Card.TFrame'
        )
        log_frame.pack(fill=tk.BOTH, pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            width=60,
            height=8,
            font=('Consolas', 9),
            bg="#161622",
            fg="#e0e0e0",
            insertbackground="white",
            bd=0,
            highlightthickness=0,
            relief='flat'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')

    def create_modern_stat_card(self, parent, title, stat_name, color):
        """Create a modern stat card component"""
        card = ttk.Frame(parent, style='Card.TFrame', padding=(15, 10))
        
        # Header with icon and title
        header_frame = ttk.Frame(card)
        header_frame.pack(fill=tk.X)
        
        # Icon
        icon_label = ttk.Label(
            header_frame,
            text=title.split()[0],  # First part is emoji
            font=('Segoe UI Emoji', 14),
            foreground=color
        )
        icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Title
        title_text = ' '.join(title.split()[1:])  # Rest is title
        title_label = ttk.Label(
            header_frame,
            text=title_text,
            foreground="#bdc3c7",
            font=('Segoe UI', 10, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        # Value with modern typography
        value_frame = ttk.Frame(card)
        value_frame.pack(fill=tk.X, pady=(5, 0))
        
        value_label = ttk.Label(
            value_frame,
            text="0",
            font=('Segoe UI', 24, 'bold'),
            foreground=color
        )
        value_label.pack(side=tk.LEFT)
        
        # Unit label if needed
        if stat_name == "accuracy":
            unit_label = ttk.Label(
                value_frame,
                text="%",
                font=('Segoe UI', 14),
                foreground=color
            )
            unit_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Store reference to update later
        setattr(self, f"{stat_name}_label", value_label)
        
        # Add subtle separator
        separator = ttk.Separator(card, orient='horizontal')
        separator.pack(fill=tk.X, pady=(10, 0))
        
        return card

    def create_history_tab(self):
        """Create content for historical data tab with modern layout"""
        # Main container for history tab
        history_container = ttk.Frame(self.history_tab)
        history_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - filters (25% width)
        left_panel = ttk.Frame(history_container)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Filter Panel with modern card design
        filter_frame = ttk.LabelFrame(
            left_panel, 
            text="DATA FILTERS", 
            padding=15,
            style='Card.TFrame'
        )
        filter_frame.pack(fill=tk.X)
        
        # Date range filter
        ttk.Label(filter_frame, text="Date Range:", style='TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        date_frame = ttk.Frame(filter_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        self.start_date_entry = ttk.Entry(date_frame, style='TEntry')
        self.start_date_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.start_date_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        
        ttk.Label(date_frame, text="to", style='TLabel').pack(side=tk.LEFT)
        
        self.end_date_entry = ttk.Entry(date_frame, style='TEntry')
        self.end_date_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.end_date_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        
        # Gender filter
        ttk.Label(filter_frame, text="Gender:", style='TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        self.gender_var = tk.StringVar(value="All")
        gender_frame = ttk.Frame(filter_frame)
        gender_frame.pack(fill=tk.X)
        
        ttk.Radiobutton(
            gender_frame, 
            text="All", 
            variable=self.gender_var, 
            value="All",
            style='TRadiobutton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            gender_frame, 
            text="Male", 
            variable=self.gender_var, 
            value="Male",
            style='TRadiobutton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            gender_frame, 
            text="Female", 
            variable=self.gender_var, 
            value="Female",
            style='TRadiobutton'
        ).pack(side=tk.LEFT)
        
        # Age filter
        ttk.Label(filter_frame, text="Age Range:", style='TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        self.age_var = tk.StringVar(value="All")
        age_frame = ttk.Frame(filter_frame)
        age_frame.pack(fill=tk.X)
        
        ttk.Radiobutton(
            age_frame, 
            text="All", 
            variable=self.age_var, 
            value="All",
            style='TRadiobutton'
        ).pack(anchor=tk.W)
        
        age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        for age_range in age_ranges:
            ttk.Radiobutton(
                filter_frame, 
                text=age_range, 
                variable=self.age_var, 
                value=age_range,
                style='TRadiobutton'
            ).pack(anchor=tk.W)
        
        # Apply filters button
        apply_button = ttk.Button(
            filter_frame,
            text="Apply Filters",
            command=self.update_history_view,
            style='TButton'
        )
        apply_button.pack(fill=tk.X, pady=(15, 0), ipady=5)
        
        # Right panel - charts and data (75% width)
        right_panel = ttk.Frame(history_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Notebook for charts with modern styling
        self.charts_notebook = ttk.Notebook(right_panel)
        self.charts_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Gender analysis tab
        gender_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(gender_tab, text="  Gender Analysis  ")
        
        # Gender distribution pie chart
        pie_frame = ttk.LabelFrame(
            gender_tab, 
            text="Gender Distribution", 
            padding=10,
            style='Card.TFrame'
        )
        pie_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pie_fig, self.pie_ax = plt.subplots(figsize=(5, 4), facecolor='#161622')
        self.pie_ax.set_facecolor('#161622')
        self.pie_canvas = FigureCanvasTkAgg(self.pie_fig, master=pie_frame)
        self.pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Time series chart
        ts_frame = ttk.LabelFrame(
            gender_tab, 
            text="Detection Over Time", 
            padding=10,
            style='Card.TFrame'
        )
        ts_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ts_fig, self.ts_ax = plt.subplots(figsize=(5, 4), facecolor='#161622')
        self.ts_ax.set_facecolor('#161622')
        self.ts_canvas = FigureCanvasTkAgg(self.ts_fig, master=ts_frame)
        self.ts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Age analysis tab
        age_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(age_tab, text="  Age Analysis  ")
        
        # Age distribution bar chart
        age_frame = ttk.LabelFrame(
            age_tab, 
            text="Age Distribution", 
            padding=10,
            style='Card.TFrame'
        )
        age_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.age_fig, self.age_ax = plt.subplots(figsize=(10, 4), facecolor='#161622')
        self.age_ax.set_facecolor('#161622')
        self.age_canvas = FigureCanvasTkAgg(self.age_fig, master=age_frame)
        self.age_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Data table with modern styling
        table_frame = ttk.LabelFrame(
            right_panel, 
            text="Detection Records", 
            padding=10,
            style='Card.TFrame'
        )
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        columns = ("Timestamp", "Gender", "Confidence", "Age", "Session")
        self.history_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=8,
            style='Treeview'
        )
        
        # Configure columns
        self.history_tree.heading("Timestamp", text="Timestamp", anchor=tk.W)
        self.history_tree.heading("Gender", text="Gender", anchor=tk.W)
        self.history_tree.heading("Confidence", text="Confidence", anchor=tk.W)
        self.history_tree.heading("Age", text="Age", anchor=tk.W)
        self.history_tree.heading("Session", text="Session", anchor=tk.W)
        
        self.history_tree.column("Timestamp", width=150, anchor=tk.W)
        self.history_tree.column("Gender", width=80, anchor=tk.W)
        self.history_tree.column("Confidence", width=100, anchor=tk.W)
        self.history_tree.column("Age", width=80, anchor=tk.W)
        self.history_tree.column("Session", width=120, anchor=tk.W)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.history_tree.yview
        )
        self.history_tree.configure(yscroll=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure tags for alternating row colors
        self.history_tree.tag_configure('oddrow', background='#1e1e2d')
        self.history_tree.tag_configure('evenrow', background='#161622')

    # [Previous methods remain unchanged...]
    # All the existing methods (load_models, detect_faces, predict_gender_age, etc.)
    # should be kept exactly as they were, only the UI creation methods have been updated
    



    def load_models(self):
        """Load all required models with comprehensive error handling"""
        self.log_message("Starting model loading process...")
        
        # Try loading face detection models first
        if not self.load_face_detection_models():
            self.log_message("Failed to load face detection models", "error")
            messagebox.showerror(
                "Model Error", 
                "Critical error loading face detection models.\n"
                "Please verify model files exist in the correct locations."
            )
            return False
        
        # Then try loading gender and age models
        if not self.load_gender_age_models():
            self.log_message("Failed to load gender/age models", "error")
            messagebox.showwarning(
                "Model Warning", 
                "Gender/age models failed to load.\n"
                "Basic face detection will still work, but gender/age detection will be disabled."
            )
            self.gender_detection_enabled = False
            self.age_detection_enabled = False
        
        self.log_message("All available models loaded successfully", "success")
        return True

    def load_face_detection_models(self):
        """Try multiple face detection models with fallback"""
        try:
            # Try Haar Cascade first (fastest)
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                if not self.face_cascade.empty():
                    self.face_detection_model = "haarcascade"
                    self.log_message("Loaded Haar Cascade face detection model")
                    return True
                self.log_message("Haar Cascade model is empty", "warning")
            except Exception as e:
                self.log_message(f"Haar Cascade load failed: {str(e)}", "warning")
            
            # Try OpenCV DNN as fallback (more accurate)
            try:
                self.face_net = cv2.dnn.readNet(
                    "models/opencv_face_detector_uint8.pb",
                    "models/opencv_face_detector.pbtxt"
                )
                self.face_detection_model = "dnn"
                self.log_message("Loaded OpenCV DNN face detection model")
                return True
            except Exception as e:
                self.log_message(f"OpenCV DNN load failed: {str(e)}", "error")
            
            raise Exception("All face detection model loading methods failed")
            
        except Exception as e:
            self.log_message(f"Error loading face detection models: {str(e)}", "error")
            traceback.print_exc()
            return False

    def load_gender_age_models(self):
        """Try multiple loading methods for gender and age models"""
        try:
            # Try OpenCV DNN models first (more reliable)
            try:
                # Gender model
                self.gender_net = cv2.dnn.readNetFromCaffe(
                    "models/gender_deploy.prototxt",
                    "models/gender_net.caffemodel"
                )
                
                # Age model
                self.age_net = cv2.dnn.readNetFromCaffe(
                    "models/age_deploy.prototxt",
                    "models/age_net.caffemodel"
                )
                self.log_message("Loaded gender/age models (OpenCV Caffe)")
                return True
            except Exception as e:
                self.log_message(f"OpenCV Caffe load failed: {str(e)}", "warning")
            
            # Try TensorFlow/Keras models as fallback
            try:
                from keras.models import load_model
                self.gender_model = load_model("models/gender_model.h5")
                self.age_model = load_model("models/age_model.h5")
                self.log_message("Loaded gender/age models (TensorFlow/Keras)")
                return True
            except Exception as e:
                self.log_message(f"Keras load failed: {str(e)}", "warning")
            
            raise Exception("All gender/age model loading methods failed")
            
        except Exception as e:
            self.log_message(f"Error loading gender/age models: {str(e)}", "error")
            traceback.print_exc()
            return False
            
    def toggle_recording(self):
        """Toggle video recording on/off"""
        if not self.recording:
            # Start recording
            file_path = filedialog.asksaveasfilename(
                defaultextension=".avi",
                filetypes=[("AVI Files", "*.avi"), ("MP4 Files", "*.mp4")],
                title="Save Recording As"
            )
            
            if file_path:
                # Get frame size from current frame
                if self.current_frame is not None:
                    height, width = self.current_frame.shape[:2]
                    
                    # Initialize video writer
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.output_video = cv2.VideoWriter(
                        file_path,
                        fourcc,
                        20.0,
                        (width, height)
                    )
                    
                    self.recording = True
                    self.record_button.config(text="⏹ Stop Recording")
                    self.recording_indicator.config(text="REC")
                    self.log_message(f"Started recording to {file_path}")
        else:
            # Stop recording
            if self.output_video:
                self.output_video.release()
                self.output_video = None
                
            self.recording = False
            self.record_button.config(text="🔴 Start Recording")
            self.recording_indicator.config(text="")
            self.log_message("Recording stopped")

    def show_help(self):
        """Show help information in a modern dialog"""
        help_text = """
        VisionAI Gender & Age Analyzer Pro - Help
        
        * Real-time Analysis Tab *
        - Click 'Start Camera' to begin live gender and age detection
        - Click 'Stop Camera' to stop the camera feed
        - Use 'Upload Video' to analyze a recorded video file
        - Click 'Start Recording' to save the camera feed to a file
        
        * Historical Data Tab *
        - View statistics and charts of previous detections
        - Filter data by date range, gender and age
        - Export data to CSV for further analysis
        
        * Settings *
        - Adjust face, gender and age detection settings
        - Change camera source if multiple cameras available
        
        * Export Data *
        - Export detection history to CSV format
        
        For more information, please contact support.
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("500x400")
        help_window.resizable(False, False)
        help_window.configure(bg="#1e1e2d")
        
        # Header
        help_header = ttk.Frame(help_window, style='Card.TFrame')
        help_header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            help_header,
            text="Help Center",
            font=('Segoe UI', 14, 'bold'),
            style='Title.TLabel'
        ).pack(pady=5)
        
        # Content
        text_widget = scrolledtext.ScrolledText(
            help_window,
            wrap=tk.WORD,
            width=60,
            height=20,
            font=('Segoe UI', 10),
            bg="#161622",
            fg="white",
            insertbackground="white",
            bd=0,
            highlightthickness=0,
            relief='flat'
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        text_widget.insert(tk.INSERT, help_text)
        text_widget.configure(state='disabled')
        
        # Close button
        close_button = ttk.Button(
            help_window,
            text="Close",
            command=help_window.destroy,
            style='TButton'
        )
        close_button.pack(pady=(0, 10), ipadx=20, ipady=5)
    


    def load_history_from_file(self):
        """Load detection history from JSON file with error handling"""
        history_file = "detection_history.json"
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.detection_history = json.load(f)
                self.log_message(f"Loaded history from {history_file} ({len(self.detection_history)} records)", 
                               "success")
                
                # Convert old format if needed (backward compatibility)
                if self.detection_history and 'timestamp' not in self.detection_history[0]:
                    self.log_message("Converting old history format", "warning")
                    for entry in self.detection_history:
                        if 'time' in entry:
                            entry['timestamp'] = entry.pop('time')
                        if 'date' in entry:
                            entry['timestamp'] = f"{entry.pop('date')} {entry.get('timestamp', '00:00:00')}"
                        entry['session'] = entry.get('session', 'Legacy Import')
                    self.save_history_to_file()
                
                return True
            else:
                self.detection_history = []
                self.log_message("No history file found - starting fresh", "info")
                return True
                
        except json.JSONDecodeError as e:
            self.log_message(f"Invalid JSON in history file: {str(e)}", "error")
            # Try to recover by backing up corrupt file
            try:
                os.rename(history_file, f"{history_file}.corrupt.{time.time()}")
                self.log_message(f"Backed up corrupt file as {history_file}.corrupt", "warning")
            except:
                pass
            self.detection_history = []
            return False
            
        except Exception as e:
            self.log_message(f"Error loading history: {str(e)}", "error")
            traceback.print_exc()
            self.detection_history = []
            return False

    def save_history_to_file(self):
        """Save detection history to JSON file with error handling"""
        history_file = "detection_history.json"
        try:
            # Create backup of current file if it exists
            if os.path.exists(history_file):
                os.replace(history_file, f"{history_file}.bak")
                
            # Save new file
            with open(history_file, 'w') as f:
                json.dump(self.detection_history, f, indent=4)
                
            self.log_message(f"Saved {len(self.detection_history)} records to {history_file}", "info")
            return True
            
        except Exception as e:
            self.log_message(f"Error saving history: {str(e)}", "error")
            traceback.print_exc()
            
            # Try to restore backup if save failed
            if os.path.exists(f"{history_file}.bak"):
                try:
                    os.replace(f"{history_file}.bak", history_file)
                    self.log_message("Restored previous history file from backup", "warning")
                except:
                    self.log_message("Failed to restore backup!", "error")
                    
            return False
        

    def open_settings(self):
        """Open the modern settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("450x500")
        settings_window.resizable(False, False)
        settings_window.configure(bg="#1e1e2d")
        
        # Header
        settings_header = ttk.Frame(settings_window, style='Card.TFrame')
        settings_header.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(
            settings_header,
            text="Application Settings",
            font=('Segoe UI', 14, 'bold'),
            style='Title.TLabel'
        ).pack(pady=5)
        
        # Content frame
        content_frame = ttk.Frame(settings_window, style='Card.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Detection settings
        ttk.Label(
            content_frame,
            text="Detection Settings:",
            font=('Segoe UI', 10, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W, pady=(10, 5), padx=10)
        
        # Face detection toggle
        face_detection_var = tk.BooleanVar(value=self.face_detection_enabled)
        face_detection_cb = ttk.Checkbutton(
            content_frame,
            text="Enable Face Detection",
            variable=face_detection_var,
            command=lambda: self.toggle_face_detection(face_detection_var.get()),
            style='TCheckbutton'
        )
        face_detection_cb.pack(pady=5, padx=20, anchor=tk.W)
        
        # Gender detection toggle
        gender_detection_var = tk.BooleanVar(value=self.gender_detection_enabled)
        gender_detection_cb = ttk.Checkbutton(
            content_frame,
            text="Enable Gender Detection",
            variable=gender_detection_var,
            command=lambda: self.toggle_gender_detection(gender_detection_var.get()),
            style='TCheckbutton'
        )
        gender_detection_cb.pack(pady=5, padx=20, anchor=tk.W)
        
        # Age detection toggle
        age_detection_var = tk.BooleanVar(value=self.age_detection_enabled)
        age_detection_cb = ttk.Checkbutton(
            content_frame,
            text="Enable Age Detection",
            variable=age_detection_var,
            command=lambda: self.toggle_age_detection(age_detection_var.get()),
            style='TCheckbutton'
        )
        age_detection_cb.pack(pady=5, padx=20, anchor=tk.W)
        
        # Model selection
        ttk.Label(
            content_frame,
            text="Face Detection Model:",
            font=('Segoe UI', 10, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W, pady=(10, 5), padx=10)
        
        model_var = tk.StringVar(value=self.face_detection_model)
        model_frame = ttk.Frame(content_frame)
        model_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Radiobutton(
            model_frame,
            text="Haar Cascade (Faster)",
            variable=model_var,
            value="haarcascade",
            style='TRadiobutton'
        ).pack(anchor=tk.W)
        
        ttk.Radiobutton(
            model_frame,
            text="OpenCV DNN (More Accurate)",
            variable=model_var,
            value="dnn",
            style='TRadiobutton'
        ).pack(anchor=tk.W)
        
        # Camera settings
        ttk.Label(
            content_frame,
            text="Camera Settings:",
            font=('Segoe UI', 10, 'bold'),
            style='TLabel'
        ).pack(anchor=tk.W, pady=(10, 5), padx=10)
        
        ttk.Label(
            content_frame,
            text="Camera Index:",
            style='TLabel'
        ).pack(anchor=tk.W, padx=20)
        
        camera_var = tk.IntVar(value=self.camera_index)
        camera_spinbox = ttk.Spinbox(
            content_frame,
            from_=0,
            to=10,
            textvariable=camera_var,
            style='TEntry'
        )
        camera_spinbox.pack(pady=5, padx=20, anchor=tk.W)
        
        # Save button
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        save_button = ttk.Button(
            button_frame,
            text="Save Settings",
            command=lambda: self.save_settings(camera_var.get(), model_var.get()),
            style='TButton'
        )
        save_button.pack(pady=10, ipadx=20, ipady=5)
        
        # Close button
        close_button = ttk.Button(
            button_frame,
            text="Close",
            command=settings_window.destroy,
            style='TButton'
        )
        close_button.pack(ipadx=20, ipady=5)

    # [Rest of the methods remain unchanged...]
    # All other methods from the original code should be kept as they were
    def export_data(self):
        """Export detection data to CSV with modern file dialog"""
        # Save current history first
        self.save_history_to_file()
        
        if not self.detection_history:
            messagebox.showwarning("No Data", "No detection data to export")
            return
            
        # Create modern file dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[
                ("CSV Files", "*.csv"), 
                ("Excel Files", "*.xlsx"),
                ("JSON Files", "*.json"),
                ("All Files", "*.*")
            ],
            title="Save Detection Data As",
            initialfile=f"detection_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.detection_history)
                
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                elif file_path.endswith('.json'):
                    df.to_json(file_path, indent=4)
                else:
                    df.to_csv(file_path, index=False)  # Default to CSV
                
                self.log_message(f"Data exported successfully to {file_path}")
                
                # Show success notification
                success_window = tk.Toplevel(self.root)
                success_window.title("Export Successful")
                success_window.geometry("300x150")
                success_window.resizable(False, False)
                success_window.configure(bg="#1e1e2d")
                
                ttk.Label(
                    success_window,
                    text="✓ Data Exported",
                    font=('Segoe UI', 14, 'bold'),
                    foreground="#2ecc71",
                    background="#1e1e2d"
                ).pack(pady=20)
                
                ttk.Label(
                    success_window,
                    text=file_path,
                    font=('Segoe UI', 9),
                    foreground="#bdc3c7",
                    background="#1e1e2d",
                    wraplength=280
                ).pack(pady=5)
                
                ttk.Button(
                    success_window,
                    text="OK",
                    command=success_window.destroy,
                    style='TButton'
                ).pack(pady=10)
                
            except Exception as e:
                self.log_message(f"Export error: {str(e)}")
                messagebox.showerror(
                    "Export Error",
                    f"Failed to export data:\n{str(e)}",
                    parent=self.root
                )
    

    def update_history_view(self):
        """Update the history view with filtered data and refresh charts"""
        try:
            # Clear current items in the treeview
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Get filter values
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            gender_filter = self.gender_var.get()
            age_filter = self.age_var.get()
            
            # Filter data based on selections
            filtered_data = []
            for entry in self.detection_history:
                try:
                    entry_date = datetime.datetime.strptime(entry['timestamp'], "%Y-%m-%d %H:%M:%S").date()
                    start_date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
                    end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
                    
                    date_ok = start_date_obj <= entry_date <= end_date_obj
                    gender_ok = (gender_filter == "All") or (entry.get('gender', 'Unknown') == gender_filter)
                    age_ok = (age_filter == "All") or (entry.get('age', 'Unknown') == age_filter)
                    
                    if date_ok and gender_ok and age_ok:
                        filtered_data.append(entry)
                except Exception as e:
                    print(f"Error filtering entry: {e}")
                    continue
            
            # Add filtered items to treeview with alternating colors
            for i, entry in enumerate(filtered_data):
                tag = 'evenrow' if i % 2 == 0 else 'oddrow'
                self.history_tree.insert("", tk.END, values=(
                    entry['timestamp'],
                    entry.get('gender', 'N/A'),
                    f"{entry.get('gender_confidence', 'N/A'):.1f}%" if isinstance(entry.get('gender_confidence'), (int, float)) else 'N/A',
                    entry.get('age', 'N/A'),
                    entry.get('session', 'N/A')
                ), tags=(tag,))
            
            # Update all charts with filtered data
            self.update_charts(filtered_data)
            self.update_age_chart()
            
            # Update status
            self.status_label.config(text=f"Showing {len(filtered_data)} records")
            
        except Exception as e:
            print(f"Error updating history view: {e}")
            self.log_message(f"Error updating history view: {e}")

    def update_charts(self, data):
        """Update the gender distribution and time series charts"""
        try:
            # Clear previous charts
            self.pie_ax.clear()
            self.ts_ax.clear()
            
            if data:
                # Pie chart - Gender distribution
                genders = [entry.get('gender', 'Unknown') for entry in data]
                male_count = genders.count("Male")
                female_count = genders.count("Female")
                other_count = len(genders) - male_count - female_count
                
                labels = []
                sizes = []
                colors = []
                
                if male_count > 0:
                    labels.append('Male')
                    sizes.append(male_count)
                    colors.append('#3498db')
                if female_count > 0:
                    labels.append('Female')
                    sizes.append(female_count)
                    colors.append('#e74c3c')
                if other_count > 0:
                    labels.append('Unknown')
                    sizes.append(other_count)
                    colors.append('#95a5a6')
                
                if sizes:  # Only draw if we have data
                    self.pie_ax.pie(sizes, labels=labels, colors=colors, 
                                   autopct='%1.1f%%', startangle=90,
                                   textprops={'color': 'white'})
                    self.pie_ax.set_title('Gender Distribution', color='white', pad=20)
                
                # Time series chart - Detections over time
                df = pd.DataFrame(data)
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['date'] = df['timestamp'].dt.date
                    
                    daily_counts = df.groupby(['date', 'gender']).size().unstack(fill_value=0)
                    
                    if 'Male' in daily_counts.columns:
                        self.ts_ax.plot(daily_counts.index, daily_counts['Male'], 
                                      label='Male', color='#3498db', marker='o')
                    if 'Female' in daily_counts.columns:
                        self.ts_ax.plot(daily_counts.index, daily_counts['Female'], 
                                      label='Female', color='#e74c3c', marker='o')
                    
                    self.ts_ax.set_title('Detections Over Time', color='white', pad=20)
                    self.ts_ax.set_xlabel('Date', color='white')
                    self.ts_ax.set_ylabel('Count', color='white')
                    self.ts_ax.legend(facecolor='#161622', edgecolor='none', 
                                    labelcolor='white')
                    self.ts_ax.tick_params(colors='white')
                    self.ts_ax.grid(color='#3a3a5d', linestyle='--')
                    
                    # Rotate x-axis labels for better readability
                    plt.setp(self.ts_ax.get_xticklabels(), rotation=45, ha="right")
                    
                    # Format x-axis as dates
                    self.ts_ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
                except Exception as e:
                    print(f"Error creating time series: {e}")
            
            # Set dark backgrounds for charts
            self.pie_ax.set_facecolor('#161622')
            self.pie_fig.patch.set_facecolor('#161622')
            self.ts_ax.set_facecolor('#161622')
            self.ts_fig.patch.set_facecolor('#161622')
            
            # Redraw canvases
            self.pie_canvas.draw()
            self.ts_canvas.draw()
            
        except Exception as e:
            print(f"Error updating charts: {e}")
            self.log_message(f"Chart update error: {e}")

    def update_age_chart(self):
        """Update the age distribution bar chart"""
        try:
            self.age_ax.clear()
            
            if not self.age_distribution:
                self.age_ax.text(0.5, 0.5, 'No age data available', 
                                ha='center', va='center', color='white')
            else:
                # Prepare age data
                age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                             '(25-32)', '(38-43)', '(48-53)', '(60-100)']
                
                # Ensure all age ranges are in the distribution, even with 0 counts
                full_dist = {age: self.age_distribution.get(age, 0) for age in age_ranges}
                
                # Filter out age ranges with 0 counts if there are many
                if sum(1 for count in full_dist.values() if count > 0) > 5:
                    filtered_dist = {k: v for k, v in full_dist.items() if v > 0}
                else:
                    filtered_dist = full_dist
                
                ages = sorted(filtered_dist.keys())
                counts = [filtered_dist[age] for age in ages]
                
                # Create color gradient based on count values
                norm = plt.Normalize(min(counts), max(counts) if counts else 1)
                colors = plt.cm.viridis(norm(counts))
                
                bars = self.age_ax.bar(ages, counts, color=colors)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    self.age_ax.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(height)}',
                                    ha='center', va='bottom', 
                                    color='white', fontsize=10)
                
                self.age_ax.set_title('Age Distribution', color='white', pad=20)
                self.age_ax.set_xlabel('Age Range', color='white')
                self.age_ax.set_ylabel('Count', color='white')
                self.age_ax.tick_params(axis='x', colors='white')
                self.age_ax.tick_params(axis='y', colors='white')
                
                # Rotate x-axis labels for better readability
                plt.setp(self.age_ax.get_xticklabels(), rotation=45, ha="right")
                
                # Add grid lines
                self.age_ax.grid(axis='y', color='#3a3a5d', linestyle='--')
            
            # Set dark background
            self.age_ax.set_facecolor('#161622')
            self.age_fig.patch.set_facecolor('#161622')
            self.age_canvas.draw()
            
        except Exception as e:
            print(f"Error updating age chart: {e}")
            self.log_message(f"Age chart error: {e}")

    def show_progress(self, value, message, error=False):
        """Show progress bar with message and optional error styling"""
        try:
            self.progress_bar['value'] = value
            self.progress_label.config(text=message)
            
            if error:
                self.style.configure('Horizontal.TProgressbar', 
                                   background='#e74c3c',  # Red for errors
                                   troughcolor='#161622')
            else:
                self.style.configure('Horizontal.TProgressbar', 
                                   background='#2ecc71',  # Green for normal
                                   troughcolor='#161622')
            
            self.progress_frame.pack(fill=tk.X, pady=(10, 0))
            self.progress_label.pack()
            self.root.update_idletasks()  # Force UI update
        except Exception as e:
            print(f"Error showing progress: {e}")

    def hide_progress(self):
        """Hide the progress bar from view"""
        try:
            self.progress_frame.pack_forget()
            self.progress_label.pack_forget()
            self.root.update_idletasks()  # Force UI update
        except Exception as e:
            print(f"Error hiding progress: {e}")

    


    def create_status_bar(self):
        """Create and configure the status bar at the bottom of the window"""
        self.status_frame = ttk.Frame(
            self.main_frame, 
            style='Card.TFrame',
            padding=(10, 5, 10, 5)
        )
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status message label (left-aligned)
        self.status_label = ttk.Label(
            self.status_frame,
            text="Status: Ready",
            foreground="#bdc3c7",
            anchor=tk.W,
            font=('Segoe UI', 9)
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Progress indicator (center)
        self.status_progress = ttk.Progressbar(
            self.status_frame,
            orient=tk.HORIZONTAL,
            mode='determinate',
            length=200,
            style='Status.Horizontal.TProgressbar'
        )
        self.status_progress.pack(side=tk.LEFT, padx=10)
        self.status_progress.pack_forget()  # Hidden by default
        
        # System indicators (right-aligned)
        indicators_frame = ttk.Frame(self.status_frame)
        indicators_frame.pack(side=tk.RIGHT)
        
        # CPU/Memory indicator
        self.system_usage_label = ttk.Label(
            indicators_frame,
            text="CPU: --% | RAM: --MB",
            foreground="#95a5a6",
            font=('Segoe UI', 9)
        )
        self.system_usage_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Version info
        self.version_label = ttk.Label(
            indicators_frame,
            text="v2.2.0",
            foreground="#7f8c8d",
            font=('Segoe UI', 9)
        )
        self.version_label.pack(side=tk.LEFT)
        
        # Configure status progress bar style
        self.style.configure(
            'Status.Horizontal.TProgressbar',
            background='#2ecc71',
            troughcolor='#1e1e2d',
            bordercolor='#1e1e2d',
            lightcolor='#2ecc71',
            darkcolor='#2ecc71',
            thickness=5
        )
        
        # Start system monitoring thread
        self.start_system_monitoring()

    def start_system_monitoring(self):
        """Start a thread to monitor system resources"""
        def monitor():
            import psutil
            while getattr(self, 'monitoring', True):
                try:
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().used / (1024 * 1024)  # MB
                    self.root.after(0, self.update_system_usage, cpu, mem)
                    time.sleep(2)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    break
        
        self.monitoring = True
        threading.Thread(target=monitor, daemon=True).start()

    def update_system_usage(self, cpu, mem):
        """Update the system usage indicators"""
        self.system_usage_label.config(
            text=f"CPU: {cpu:.1f}% | RAM: {mem:.0f}MB"
        )
        
        # Change color based on load
        if cpu > 80:
            self.system_usage_label.config(foreground="#e74c3c")
        elif cpu > 50:
            self.system_usage_label.config(foreground="#f39c12")
        else:
            self.system_usage_label.config(foreground="#2ecc71")

    def show_status_progress(self, show=True):
        """Show or hide the status bar progress indicator"""
        if show:
            self.status_progress.pack(side=tk.LEFT, padx=10)
        else:
            self.status_progress.pack_forget()
        self.root.update_idletasks()

    def update_status(self, message, progress=None, error=False):
        """Update the status bar message and optional progress"""
        try:
            self.status_label.config(text=f"Status: {message}")
            
            if progress is not None:
                self.status_progress['value'] = progress
                self.show_status_progress(True)
            else:
                self.show_status_progress(False)
            
            if error:
                self.status_label.config(foreground="#e74c3c")
                self.style.configure('Status.Horizontal.TProgressbar', background='#e74c3c')
            else:
                self.status_label.config(foreground="#bdc3c7")
                self.style.configure('Status.Horizontal.TProgressbar', background='#2ecc71')
            
            self.root.update_idletasks()
        except Exception as e:
            print(f"Status update error: {e}")
    

    def verify_models(self):
        """Verify that all required model files exist and are accessible"""
        required_files = {
            "Face Detection (Haar Cascade)": [
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            ],
            "Face Detection (DNN)": [
                "models/opencv_face_detector.pbtxt",
                "models/opencv_face_detector_uint8.pb"
            ],
            "Gender Detection": [
                "models/gender_deploy.prototxt",
                "models/gender_net.caffemodel"
            ],
            "Age Detection": [
                "models/age_deploy.prototxt",
                "models/age_net.caffemodel"
            ]
        }

        missing_files = {}
        status = True
        message = "All model files present"

        # Check each model type
        for model_type, files in required_files.items():
            for file_path in files:
                if not os.path.exists(file_path):
                    status = False
                    if model_type not in missing_files:
                        missing_files[model_type] = []
                    missing_files[model_type].append(file_path)

        # Prepare detailed message if files are missing
        if not status:
            message = "Missing model files:\n"
            for model_type, files in missing_files.items():
                message += f"\n{model_type}:\n"
                for file_path in files:
                    message += f"- {file_path}\n"
            
            # Add installation instructions
            message += "\nPlease download the required model files and place them in the correct locations."
        
        # Log the verification result
        self.log_message("Model verification: " + ("Success" if status else "Failure"))
        if not status:
            self.log_message(message)
        
        return status, message
    



    def log_message(self, message, level="info"):
        """Add a timestamped message to the log console with level-based coloring"""
        try:
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            
            # Determine color based on level
            colors = {
                "info": "#bdc3c7",    # Gray
                "warning": "#f39c12", # Orange
                "error": "#e74c3c",    # Red
                "success": "#2ecc71"  # Green
            }
            color = colors.get(level.lower(), "#bdc3c7")
            
            # Enable text widget for editing
            self.log_text.configure(state='normal')
            
            # Insert the message with timestamp
            self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.log_text.insert(tk.END, f"{message}\n", level)
            
            # Apply tags for coloring
            self.log_text.tag_config("timestamp", foreground="#95a5a6")
            self.log_text.tag_config(level, foreground=color)
            
            # Disable editing and scroll to end
            self.log_text.configure(state='disabled')
            self.log_text.see(tk.END)
            
            # Also print to console for debugging
            print(f"[{timestamp}] {message}")
            
        except Exception as e:
            print(f"Error logging message: {e}")

    def clear_log(self):
        """Clear all messages from the log console"""
        try:
            self.log_text.configure(state='normal')
            self.log_text.delete(1.0, tk.END)
            self.log_text.configure(state='disabled')
        except Exception as e:
            print(f"Error clearing log: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GenderAgeDetectionApp(root)
    root.mainloop()
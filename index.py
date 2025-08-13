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
        self.setup_window() 
        
        # Verify models before loading
        status, message = self.verify_models()
        if not status:
            messagebox.showwarning("Model Files Missing", message)
        
        self.load_models()
        self.history = []
        self.load_history_from_file()
        self.update_history_view()
        

    def setup_window(self):
        """Configure window with dark theme"""
        self.root.title("VisionAI Gender & Age Analyzer Pro")
        self._title = "VisionAI Gender & Age Analyzer Pro"  # Add this line
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.configure(bg='#202020')  # Dark background
        

        
        # Custom dark theme style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color definitions
        dark_bg = '#202020'       # Light black background
        darker_bg = '#181818'     # Darker elements
        accent = '#3A3A3A'        # Accent color
        text_color = '#E0E0E0'    # Light text
        highlight = '#404040'     # Highlight color
        

        
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
                            borderwidth=1)
        
        self.style.map('TButton',
                    background=[('active', highlight)],
                    foreground=[('active', text_color)])
        
        self.style.configure('Title.TLabel',
                            font=('Segoe UI', 26, 'bold'),
                            foreground=text_color)
        
        self.style.configure('Card.TFrame',
                            background=darker_bg,
                            relief=tk.RAISED,
                            borderwidth=0)
        
        self.style.configure('Stat.TLabel',
                            font=('Segoe UI', 16, 'bold'),
                            foreground=text_color)
        
        self.style.configure('TNotebook',
                            background=dark_bg)
        
        self.style.configure('TNotebook.Tab',
                            font=('Segoe UI', 10, 'bold'),
                            padding=[10, 5],
                            background=darker_bg,
                            foreground=text_color)
        
        self.style.map('TNotebook.Tab',
                    background=[('selected', accent)],
                    foreground=[('selected', text_color)])
        
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
                            borderwidth=0)
        
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
                            darkcolor=accent)
            
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
        """Create all UI components"""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.header_frame = ttk.Frame(self.main_frame, style='Card.TFrame')
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = ttk.Label(
            self.header_frame, 
            text="VisionAI Gender Analyzer", 
            style="Title.TLabel"
        )
        self.title_label.pack(side=tk.LEFT, padx=20)
        
        # Header buttons
        btn_frame = ttk.Frame(self.header_frame)
        btn_frame.pack(side=tk.RIGHT, padx=10)
        
        self.settings_button = ttk.Button(
            btn_frame,
            text="⚙️ Settings",
            command=self.open_settings,
            style='TButton'
        )
        self.settings_button.pack(side=tk.LEFT, padx=5)
        
        self.help_button = ttk.Button(
            btn_frame,
            text="❓ Help",
            command=self.show_help,
            style='TButton'
        )
        self.help_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = ttk.Button(
            btn_frame,
            text="📊 Export Data",
            command=self.export_data,
            style='TButton'
        )
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Real-time Analysis Tab
        self.realtime_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.realtime_tab, text="Real-time Analysis")
        self.create_realtime_tab()
        
        # Historical Data Tab
        self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.history_tab, text="Historical Data")
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
            text="v2.2.0",
            foreground="#ffffff",
            anchor=tk.E
        )
        self.version_label.pack(side=tk.RIGHT, padx=10)
    
    def create_realtime_tab(self):
        """Create content for real-time analysis tab"""
        # Left panel - controls and stats
        left_panel = ttk.Frame(self.realtime_tab)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Control Panel
        control_frame = ttk.LabelFrame(
            left_panel, 
            text="Capture Controls", 
            padding=15,
            style='Card.TFrame'
        )
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Modified buttons to launch separate scripts
        self.start_button = ttk.Button(
            control_frame,
            text="🎥 Start Camera",
            command=lambda: self.launch_script("start_camera.py"),
            style='TButton'
        )
        self.start_button.pack(fill=tk.X, pady=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="⏹ Stop Camera",
            command=lambda: self.launch_script("stop_camera.py"),
            state=tk.NORMAL,  # Changed to normal for demo
            style='TButton'
        )
        self.stop_button.pack(fill=tk.X, pady=5)
        
        self.upload_button = ttk.Button(
            control_frame,
            text="📁 Upload Video",
            command=self.open_video_upload_window,
            style='TButton'
        )

        self.upload_button.pack(fill=tk.X, pady=5)
    
        
        
        
        # Stats Panel
        stats_frame = ttk.LabelFrame(
            left_panel, 
            text="Detection Statistics", 
            padding=5,
            style='Card.TFrame'
        )
        stats_frame.pack(fill=tk.X)
        
        # Create stat cards
        self.male_card = self.create_stat_card(stats_frame, "Male", "#ffffff", "👨")
        self.male_card.pack(fill=tk.X, pady=5)
        
        self.female_card = self.create_stat_card(stats_frame, "Female", "#e74c3c", "👩")
        self.female_card.pack(fill=tk.X, pady=5)
        
        self.fps_card = self.create_stat_card(stats_frame, "FPS", "#2ecc71", "⏱")
        self.fps_card.pack(fill=tk.X, pady=5)
        
        self.accuracy_card = self.create_stat_card(stats_frame, "Accuracy", "#9b59b6", "🎯")
        self.accuracy_card.pack(fill=tk.X, pady=5)
        
        self.duration_card = self.create_stat_card(stats_frame, "Duration", "#f39c12", "⏳")
        self.duration_card.pack(fill=tk.X, pady=5)
        
        # Right panel - display and logs
        right_panel = ttk.Frame(self.realtime_tab)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Display frame
        display_frame = ttk.LabelFrame(
            right_panel, 
            text="Live Feed", 
            padding=0,
            style='Card.TFrame'
        )
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            display_frame, 
            bg="#ffffff", 
            bd=0,
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_frame = ttk.Frame(right_panel, style='Card.TFrame')
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            orient=tk.HORIZONTAL,
            length=500,
            mode='determinate',
            style='Progress.Horizontal.TProgressbar'
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_label = ttk.Label(
            self.progress_frame,
            text="Ready",
            foreground="#bdc3c7"
        )
        self.progress_label.pack(pady=(0, 5))
        
        self.hide_progress()
        
        # Log console
        log_frame = ttk.LabelFrame(
            right_panel, 
            text="Detection Log", 
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
            bg="#bebfc0",
            fg="white",
            insertbackground="white"
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state='disabled')
    
    def create_stat_card(self, parent, title, color, icon=None):
        """Create a stat card component"""
        card = ttk.Frame(parent, style='Card.TFrame', padding=10)
        
        # Header
        header_frame = ttk.Frame(card)
        header_frame.pack(fill=tk.X)
        
        if icon:
            icon_label = ttk.Label(
                header_frame,
                text=icon,
                font=('Segoe UI Emoji', 14),
                foreground=color
            )
            icon_label.pack(side=tk.LEFT, padx=(0, 5))
        
        title_label = ttk.Label(
            header_frame,
            text=title,
            foreground=color,
            font=('Segoe UI', 11, 'bold')
        )
        title_label.pack(side=tk.LEFT)
        
        # Value
        value_label = ttk.Label(
            card,
            text="0",
            font=('Segoe UI', 24, 'bold'),
            foreground="#ecf0f1"
        )
        value_label.pack(fill=tk.X, pady=(5, 0))
        
        # Store reference to update later
        stat_name = title.lower().replace(" ", "_")
        setattr(self, f"{stat_name}_label", value_label)
        
        return card
    
    def update_status(self, message):
        """Update the status text in the status bar"""
        try:
            self.status_label.config(text=f"Status: {message}")
            self.root.update_idletasks()  # Force UI update
        except Exception as e:
            print(f"Error updating status: {e}")

    def create_status_bar(self):
        """Create the status bar at the bottom of the window"""
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
            text="v2.2.0",
            foreground="#bdc3c7",
            anchor=tk.E
        )
        self.version_label.pack(side=tk.RIGHT, padx=10)
    
    

    def load_models(self):
        """Load models with debug output"""
        try:
            print("Attempting to load face detection model...")
            
            # Try multiple face detection models
            if not self.load_face_detection_models():
                raise Exception("All face detection model loading attempts failed")
            
            # Try loading gender and age models
            if not self.load_gender_age_models():
                raise Exception("All gender and age model loading attempts failed")
            
            print("All models loaded successfully")
            return True
            
        except Exception as e:
            print(f"MODEL LOAD ERROR: {str(e)}")
            messagebox.showerror("Model Error", f"Failed to load models: {str(e)}")
            return False

    def load_face_detection_models(self):
        """Try multiple face detection models"""
        try:
            # Try Haar Cascade first
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    raise Exception("Haar Cascade model is empty")
                print("Loaded face detection model with Haar Cascade")
                self.face_detection_model = "haarcascade"
                return True
            except Exception as e:
                print(f"Haar Cascade load failed: {e}")
            
            # Try OpenCV DNN
            try:
                self.face_net = cv2.dnn.readNet(
                    "models/opencv_face_detector_uint8.pb",
                    "models/opencv_face_detector.pbtxt"
                )
                print("Loaded face detection model with OpenCV DNN")
                self.face_detection_model = "dnn"
                return True
            except Exception as e:
                print(f"OpenCV DNN load failed: {e}")
                
            raise Exception("All face detection model loading methods failed")
            
        except Exception as e:
            print(f"Error loading face detection model: {e}")
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
                print("Loaded gender and age models with OpenCV (Caffe format)")
                return True
            except Exception as e:
                print(f"OpenCV Caffe load failed: {e}")
            
            # Try TensorFlow/Keras models as fallback
            try:
                from keras.models import load_model
                self.gender_model = load_model("models/gender_model.h5")
                self.age_model = load_model("models/age_model.h5")
                print("Loaded gender and age models with TensorFlow/Keras")
                return True
            except Exception as e:
                print(f"Keras load failed: {e}")
                
            raise Exception("All model loading methods failed")
            
        except Exception as e:
            print(f"Error loading gender and age models: {e}")
            return False

    def detect_faces(self, frame):
        """Detect faces using the current face detection model"""
        if self.face_detection_model == "haarcascade":
            # Convert to grayscale for Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return [(x, y, x+w, y+h) for (x, y, w, h) in faces]
        else:  # DNN model
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (300, 300), 
                [104, 117, 123], swapRB=False, crop=False
            )
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:  # Confidence threshold
                    box = detections[0, 0, i, 3:7] * np.array([
                        frame.shape[1], frame.shape[0], 
                        frame.shape[1], frame.shape[0]
                    ])
                    (x1, y1, x2, y2) = box.astype("int")
                    faces.append((x1, y1, x2, y2))
            return faces

    def predict_gender_age(self, face_img):
        """Predict gender and age with better error handling"""
        try:
            if face_img.size == 0:
                print("Empty face image!")
                return "Unknown", 0.0, "Unknown"
                
            # Try OpenCV DNN first (more reliable)
            if hasattr(self, 'gender_net') and hasattr(self, 'age_net'):
                print("Using OpenCV DNN for prediction")
                
                # Preprocess for gender detection
                gender_blob = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                self.gender_net.setInput(gender_blob)
                gender_preds = self.gender_net.forward()
                gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"
                gender_confidence = max(gender_preds[0][0], gender_preds[0][1]) * 100
                
                # Preprocess for age detection
                age_blob = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                self.age_net.setInput(age_blob)
                age_preds = self.age_net.forward()
                age = self.age_list[age_preds[0].argmax()]
                
                return gender, gender_confidence, age
                
            # Fall back to Keras models if available
            elif hasattr(self, 'gender_model') and hasattr(self, 'age_model'):
                print("Using Keras models for prediction")
                img = cv2.resize(face_img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Predict gender
                gender_pred = self.gender_model.predict(img)
                gender = "Male" if gender_pred[0][0] > 0.5 else "Female"
                gender_confidence = float(np.max(gender_pred)) * 100
                
                # Predict age
                age_pred = self.age_model.predict(img)
                age = str(int(age_pred[0][0]))
                
                return gender, gender_confidence, age
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            traceback.print_exc()
            return "Unknown", 0.0, "Unknown"

    def process_frame(self, frame):
        """Process each frame for face detection, gender and age prediction"""
        try:
            # Only process every few frames to improve performance
            current_time = time.time()
            if current_time - self.last_processed_time < self.processing_interval:
                self.update_display(frame)
                return
            
            self.last_processed_time = current_time
            
            # Detect faces
            faces = self.detect_faces(frame) if self.face_detection_enabled else []
            
            # Reset counts for this frame
            current_frame_persons = 0
            
            # Process each face
            for (x1, y1, x2, y2) in faces:
                # Limit to 5 persons per frame
                if current_frame_persons >= 5:
                    break
                    
                current_frame_persons += 1
                
                # Extract face ROI
                face_img = frame[y1:y2, x1:x2]
                
                # Predict gender and age if enabled
                gender = "Unknown"
                age = "Unknown"
                confidence = 0
                
                if self.gender_detection_enabled or self.age_detection_enabled:
                    gender, confidence, age = self.predict_gender_age(face_img)
                    
                    # Update counts
                    if gender == "Male":
                        self.male_count += 1
                    elif gender == "Female":
                        self.female_count += 1
                        
                    # Update age distribution
                    if age != "Unknown":
                        self.age_distribution[age] = self.age_distribution.get(age, 0) + 1
                
                # Draw bounding box and labels
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{gender} {confidence:.1f}% - {age}"
                cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add to detection history
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.detection_history.append({
                    'timestamp': timestamp,
                    'gender': gender,
                    'gender_confidence': confidence,
                    'age': age,
                    'session': 'Live Camera'
                })
            
            # Update statistics
            self.update_stat("male", self.male_count)
            self.update_stat("female", self.female_count)
            self.update_stat("accuracy", f"{self.calculate_accuracy():.1f}%")
            
            # Log frame processing
            self.log_message(f"Processed frame with {current_frame_persons} persons detected")
            
            # Update display
            self.update_display(frame)
            
        except Exception as e:
            self.log_message(f"Error processing frame: {str(e)}")
            print(f"Error processing frame: {e}")
            traceback.print_exc()

    def capture_frames(self):
        """Capture frames from the camera with smooth playback"""
        if not self.is_capturing:
            return
            
        start_time = time.time()
        ret, frame = self.capture.read()
        
        if ret:
            self.current_frame = frame
            self.total_frames += 1
            
            # Process frame (with person limit)
            self.process_frame(frame)
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
            self.update_stat("fps", f"{fps:.1f}")
            
            # Update session duration
            if self.session_start_time:
                duration = time.time() - self.session_start_time
                self.update_stat("duration", self.format_duration(duration))
            
            # If recording, write frame to output video
            if self.recording and self.output_video:
                self.output_video.write(frame)
            
            # Schedule next frame capture with dynamic delay for smooth playback
            target_fps = 30  # Adjust as needed
            delay = max(1, int(1000/target_fps - elapsed_time*1000))
            self.after_id = self.root.after(delay, self.capture_frames)
        else:
            self.update_status("Error reading frame")
            self.stop_capture()

    def update_display(self, frame):
        """Update the displayed image in the UI"""
        try:
            # Convert OpenCV BGR image to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 0 and canvas_height > 0:
                # Convert to PIL Image
                img = Image.fromarray(rgb_image)
                
                # Calculate aspect ratio
                img_ratio = img.width / img.height
                canvas_ratio = canvas_width / canvas_height
                
                # Resize while maintaining aspect ratio
                if img_ratio > canvas_ratio:
                    new_width = canvas_width
                    new_height = int(canvas_width / img_ratio)
                else:
                    new_height = canvas_height
                    new_width = int(canvas_height * img_ratio)
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage
                self.tk_image = ImageTk.PhotoImage(image=img)
                
                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor=tk.CENTER,
                    image=self.tk_image
                )
            
        except Exception as e:
            print(f"Error updating display: {e}")

    def update_stat(self, stat_name, value):
        """Update a statistic display in the UI"""
        try:
            # Find the correct label widget based on stat_name
            label_name = f"{stat_name.lower()}_label"
            if hasattr(self, label_name):
                getattr(self, label_name).config(text=str(value))
            else:
                print(f"Warning: No stat display found for {stat_name}")
        except Exception as e:
            print(f"Error updating stat {stat_name}: {str(e)}")

    def show_progress(self, value, message, error=False):
        """Show progress bar with message"""
        self.progress_bar['value'] = value
        self.progress_label.config(text=message)
        
        if error:
            self.style.configure('Progress.Horizontal.TProgressbar', background='red')
        else:
            self.style.configure('Progress.Horizontal.TProgressbar', background='green')
        
        self.progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.progress_label.pack()
    
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_frame.pack_forget()
        self.progress_label.pack_forget()

    
    
    def stop_capture(self):
        """Stop the camera capture"""
        self.is_capturing = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.NORMAL)
        self.update_status("Capture stopped")
        
        if self.session_start_time:
            duration = time.time() - self.session_start_time
            self.log_message(f"Camera session ended. Duration: {self.format_duration(duration)}")
            self.session_start_time = None
    
    
    

    def format_duration(self, seconds):
        """Format duration in seconds to HH:MM:SS"""
        return time.strftime('%H:%M:%S', time.gmtime(seconds))

    def calculate_accuracy(self):
        """Calculate detection accuracy"""
        total_detections = self.male_count + self.female_count
        if self.total_frames == 0:
            return 0.0
        return (total_detections / self.total_frames) * 100
    
    def get_session_duration(self):
        """Get current session duration"""
        if not self.session_start_time:
            return "00:00:00"
        duration = time.time() - self.session_start_time
        return self.format_duration(duration)

    def log_message(self, message):
        """Add message to log console"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.log_text.configure(state='disabled')
        self.log_text.see(tk.END)

    def load_history_from_file(self):
        """Load detection history from a JSON file"""
        history_file = "detection_history.json"
        try:
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.detection_history = json.load(f)
                self.log_message("History loaded successfully")
            else:
                self.detection_history = []
                self.log_message("No history file found - starting fresh")
        except Exception as e:
            self.detection_history = []
            self.log_message(f"Error loading history: {str(e)}")
            print(f"Error loading history: {e}")

    def save_history_to_file(self):
        """Save detection history to a JSON file"""
        history_file = "detection_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.detection_history, f, indent=4)
            self.log_message("History saved successfully")
        except Exception as e:
            self.log_message(f"Error saving history: {str(e)}")
            print(f"Error saving history: {e}")

    def create_history_tab(self):
        """Create content for historical data tab"""
        # Left panel - controls and filters
        left_panel = ttk.Frame(self.history_tab)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Filter Panel
        filter_frame = ttk.LabelFrame(
            left_panel, 
            text="Data Filters", 
            padding=15,
            style='Card.TFrame'
        )
        filter_frame.pack(fill=tk.X)
        
        ttk.Label(filter_frame, text="Date Range:").pack(anchor=tk.W)
        
        date_frame = ttk.Frame(filter_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        self.start_date_entry = ttk.Entry(date_frame)
        self.start_date_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.start_date_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        
        ttk.Label(date_frame, text="to").pack(side=tk.LEFT)
        
        self.end_date_entry = ttk.Entry(date_frame)
        self.end_date_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.end_date_entry.insert(0, datetime.date.today().strftime("%Y-%m-%d"))
        
        ttk.Label(filter_frame, text="Gender:").pack(anchor=tk.W)
        
        self.gender_var = tk.StringVar(value="All")
        ttk.Radiobutton(
            filter_frame, 
            text="All", 
            variable=self.gender_var, 
            value="All"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            filter_frame, 
            text="Male", 
            variable=self.gender_var, 
            value="Male"
        ).pack(anchor=tk.W)
        ttk.Radiobutton(
            filter_frame, 
            text="Female", 
            variable=self.gender_var, 
            value="Female"
        ).pack(anchor=tk.W)
        
        ttk.Label(filter_frame, text="Age Range:").pack(anchor=tk.W)
        
        self.age_var = tk.StringVar(value="All")
        ttk.Radiobutton(
            filter_frame, 
            text="All", 
            variable=self.age_var, 
            value="All"
        ).pack(anchor=tk.W)
        
        age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        for age_range in age_ranges:
            ttk.Radiobutton(
                filter_frame, 
                text=age_range, 
                variable=self.age_var, 
                value=age_range
            ).pack(anchor=tk.W)
        
        ttk.Button(
            filter_frame,
            text="Apply Filters",
            command=self.update_history_view,
            style='TButton'
        ).pack(fill=tk.X, pady=(10, 0))
        
        # Right panel - charts and data
        right_panel = ttk.Frame(self.history_tab)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Notebook for charts
        self.charts_notebook = ttk.Notebook(right_panel)
        self.charts_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Gender charts tab
        gender_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(gender_tab, text="Gender Analysis")
        
        # Gender distribution pie chart
        pie_frame = ttk.LabelFrame(
            gender_tab, 
            text="Gender Distribution", 
            padding=10,
            style='Card.TFrame'
        )
        pie_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.pie_fig, self.pie_ax = plt.subplots(figsize=(5, 4), facecolor='#2c3e50')
        self.pie_ax.set_facecolor('#2c3e50')
        self.pie_canvas = FigureCanvasTkAgg(self.pie_fig, master=pie_frame)
        self.pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Time series chart
        ts_frame = ttk.LabelFrame(
            gender_tab, 
            text="Detection Over Time", 
            padding=10,
            style='Card.TFrame'
        )
        ts_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.ts_fig, self.ts_ax = plt.subplots(figsize=(5, 4), facecolor='#2c3e50')
        self.ts_ax.set_facecolor('#2c3e50')
        self.ts_canvas = FigureCanvasTkAgg(self.ts_fig, master=ts_frame)
        self.ts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Age charts tab
        age_tab = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(age_tab, text="Age Analysis")
        
        # Age distribution bar chart
        age_frame = ttk.LabelFrame(
            age_tab, 
            text="Age Distribution", 
            padding=10,
            style='Card.TFrame'
        )
        age_frame.pack(fill=tk.BOTH, expand=True)
        
        self.age_fig, self.age_ax = plt.subplots(figsize=(10, 4), facecolor='#2c3e50')
        self.age_ax.set_facecolor('#2c3e50')
        self.age_canvas = FigureCanvasTkAgg(self.age_fig, master=age_frame)
        self.age_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Data table
        table_frame = ttk.LabelFrame(
            right_panel, 
            text="Detection Records", 
            padding=10,
            style='Card.TFrame'
        )
        table_frame.pack(fill=tk.BOTH, pady=(10, 0))
        
        columns = ("#1", "#2", "#3", "#4", "#5")
        self.history_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=8,
            style='Treeview'
        )
        
        self.history_tree.heading("#1", text="Timestamp")
        self.history_tree.heading("#2", text="Gender")
        self.history_tree.heading("#3", text="Confidence")
        self.history_tree.heading("#4", text="Age")
        self.history_tree.heading("#5", text="Session")
        
        self.history_tree.column("#1", width=150)
        self.history_tree.column("#2", width=80)
        self.history_tree.column("#3", width=100)
        self.history_tree.column("#4", width=80)
        self.history_tree.column("#5", width=120)
        
        scrollbar = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.history_tree.yview
        )
        self.history_tree.configure(yscroll=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_age_chart(self):
        """Update the age distribution chart"""
        if not hasattr(self, 'age_ax'):
            return
            
        self.age_ax.clear()
        
        if not self.age_distribution:
            self.age_ax.text(0.5, 0.5, 'No age data available', 
                           ha='center', va='center', color='white')
        else:
            ages = sorted(self.age_distribution.keys())
            counts = [self.age_distribution[age] for age in ages]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(ages)))
            bars = self.age_ax.bar(ages, counts, color=colors)
            
            for bar in bars:
                height = bar.get_height()
                self.age_ax.text(bar.get_x() + bar.get_width()/2., height,
                               '%d' % int(height),
                               ha='center', va='bottom', color='white')
            
            self.age_ax.set_title('Age Distribution', color='white')
            self.age_ax.set_xlabel('Age Range', color='white')
            self.age_ax.set_ylabel('Count', color='white')
            self.age_ax.tick_params(axis='x', colors='white')
            self.age_ax.tick_params(axis='y', colors='white')
            
            # Rotate x-axis labels for better readability
            plt.setp(self.age_ax.get_xticklabels(), rotation=45, ha="right")
        
        self.age_ax.set_facecolor('#2c3e50')
        self.age_fig.patch.set_facecolor('#2c3e50')
        self.age_canvas.draw()

    def update_history_view(self):
        """Update the history view with filtered data"""
        # Clear current items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Get filter values
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        gender_filter = self.gender_var.get()
        age_filter = self.age_var.get()
        
        # Filter data
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
            except:
                continue
        
        # Add filtered items to treeview
        for entry in filtered_data:
            self.history_tree.insert("", tk.END, values=(
                entry['timestamp'],
                entry.get('gender', 'N/A'),
                f"{entry.get('gender_confidence', 'N/A'):.1f}%" if 'gender_confidence' in entry else 'N/A',
                entry.get('age', 'N/A'),
                entry['session']
            ))
        
        # Update charts
        self.update_charts(filtered_data)
        self.update_age_chart()

    def update_charts(self, data):
        """Update the charts with filtered data"""
        # Clear previous charts
        self.pie_ax.clear()
        self.ts_ax.clear()
        
        # Pie chart - Gender distribution
        if data:
            genders = [entry.get('gender', 'Unknown') for entry in data]
            male_count = genders.count("Male")
            female_count = genders.count("Female")
            
            labels = ['Male', 'Female']
            sizes = [male_count, female_count]
            colors = ['#3498db', '#e74c3c']
            
            self.pie_ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            self.pie_ax.axis('equal')
            self.pie_ax.set_title('Gender Distribution', color='white')
            
            # Time series chart - Detections over time
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            daily_counts = df.groupby(['date', 'gender']).size().unstack(fill_value=0)
            
            if 'Male' in daily_counts.columns:
                self.ts_ax.plot(daily_counts.index, daily_counts['Male'], label='Male', color='#3498db')
            if 'Female' in daily_counts.columns:
                self.ts_ax.plot(daily_counts.index, daily_counts['Female'], label='Female', color='#e74c3c')
            
            self.ts_ax.set_title('Detections Over Time', color='white')
            self.ts_ax.set_xlabel('Date', color='white')
            self.ts_ax.set_ylabel('Count', color='white')
            self.ts_ax.legend()
            self.ts_ax.tick_params(colors='white')
            
            # Set background colors
            self.pie_ax.set_facecolor('#2c3e50')
            self.pie_fig.patch.set_facecolor('#2c3e50')
            self.ts_ax.set_facecolor('#2c3e50')
            self.ts_fig.patch.set_facecolor('#2c3e50')
            
            # Redraw canvases
            self.pie_canvas.draw()
            self.ts_canvas.draw()

    def verify_models(self):
        """Check if required model files exist"""
        required_files = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            "models/opencv_face_detector.pbtxt",
            "models/opencv_face_detector_uint8.pb",
            "models/gender_deploy.prototxt",
            "models/gender_net.caffemodel",
            "models/age_deploy.prototxt",
            "models/age_net.caffemodel"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
        return True, "All model files present"    

    def export_data(self):
        """Export detection data to CSV"""
        # Save current history first
        self.save_history_to_file()
        
        if not self.detection_history:
            messagebox.showwarning("No Data", "No detection data to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Detection Data"
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self.detection_history)
                df.to_csv(file_path, index=False)
                self.log_message(f"Data exported to {file_path}")
                messagebox.showinfo("Success", "Data exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {str(e)}")

    def show_help(self):
        """Show help information in a dialog"""
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
        help_window.configure(bg="#ffffff")
        
        text_widget = scrolledtext.ScrolledText(
            help_window,
            wrap=tk.WORD,
            width=60,
            height=20,
            font=('Segoe UI', 10),
            bg="#34495e",
            fg="white",
            insertbackground="white"
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.INSERT, help_text)
        text_widget.configure(state='disabled')
        
        close_button = ttk.Button(
            help_window,
            text="Close",
            command=help_window.destroy,
            style='TButton'
        )
        close_button.pack(pady=10)

    def open_settings(self):
        """Open the settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x400")
        settings_window.resizable(False, False)
        settings_window.configure(bg="#e2e2e2")
        
        # Face detection toggle
        face_detection_var = tk.BooleanVar(value=self.face_detection_enabled)
        face_detection_cb = ttk.Checkbutton(
            settings_window,
            text="Enable Face Detection",
            variable=face_detection_var,
            command=lambda: self.toggle_face_detection(face_detection_var.get()),
            style='TLabel'
        )
        face_detection_cb.pack(pady=10, padx=20, anchor=tk.W)
        
        # Gender detection toggle
        gender_detection_var = tk.BooleanVar(value=self.gender_detection_enabled)
        gender_detection_cb = ttk.Checkbutton(
            settings_window,
            text="Enable Gender Detection",
            variable=gender_detection_var,
            command=lambda: self.toggle_gender_detection(gender_detection_var.get()),
            style='TLabel'
        )
        gender_detection_cb.pack(pady=10, padx=20, anchor=tk.W)
        
        # Age detection toggle
        age_detection_var = tk.BooleanVar(value=self.age_detection_enabled)
        age_detection_cb = ttk.Checkbutton(
            settings_window,
            text="Enable Age Detection",
            variable=age_detection_var,
            command=lambda: self.toggle_age_detection(age_detection_var.get()),
            style='TLabel'
        )
        age_detection_cb.pack(pady=10, padx=20, anchor=tk.W)
        
        # Face detection model selection
        ttk.Label(settings_window, text="Face Detection Model:").pack(pady=(10, 0), padx=20, anchor=tk.W)
        model_var = tk.StringVar(value=self.face_detection_model)
        ttk.Radiobutton(
            settings_window,
            text="Haar Cascade",
            variable=model_var,
            value="haarcascade",
            style='TLabel'
        ).pack(padx=20, anchor=tk.W)
        ttk.Radiobutton(
            settings_window,
            text="OpenCV DNN",
            variable=model_var,
            value="dnn",
            style='TLabel'
        ).pack(padx=20, anchor=tk.W)
        
        # Camera selection
        ttk.Label(settings_window, text="Camera Index:").pack(pady=(10, 0), padx=20, anchor=tk.W)
        camera_var = tk.IntVar(value=self.camera_index)
        camera_spinbox = ttk.Spinbox(
            settings_window,
            from_=0,
            to=10,
            textvariable=camera_var
        )
        camera_spinbox.pack(pady=5, padx=20, anchor=tk.W)
        
        # Save button
        save_button = ttk.Button(
            settings_window,
            text="Save Settings",
            command=lambda: self.save_settings(camera_var.get(), model_var.get()),
            style='TButton'
        )
        save_button.pack(pady=20)

    def toggle_face_detection(self, enabled):
        """Toggle face detection on/off"""
        self.face_detection_enabled = enabled
        self.log_message(f"Face detection {'enabled' if enabled else 'disabled'}")

    def toggle_gender_detection(self, enabled):
        """Toggle gender detection on/off"""
        self.gender_detection_enabled = enabled
        self.log_message(f"Gender detection {'enabled' if enabled else 'disabled'}")

    def toggle_age_detection(self, enabled):
        """Toggle age detection on/off"""
        self.age_detection_enabled = enabled
        self.log_message(f"Age detection {'enabled' if enabled else 'disabled'}")

    def save_settings(self, camera_index, face_detection_model):
        """Save camera settings"""
        self.camera_index = camera_index
        self.face_detection_model = face_detection_model
        self.log_message(f"Camera index set to {camera_index}, face detection model: {face_detection_model}")
        messagebox.showinfo("Settings Saved", "Settings have been saved successfully")

    # Add these methods for external control
    def get_control_interface(self):
        """Returns a control interface for external modules"""
        class ControlInterface:
            def __init__(self, app):
                self.app = app
            
            def start_camera(self):
                self.app.start_capture()
            
            def stop_camera(self):
                self.app.stop_capture()
            
            def upload_video(self):
                self.app.open_file_dialog()
            
            def toggle_recording(self):
                self.app.toggle_recording()
        
        return ControlInterface(self)
    

    def launch_script(self, script_name):
        """Launch a separate Python script"""
        try:
            if not os.path.exists(script_name):
                self.log_message(f"Error: {script_name} not found!")
                return
            
            subprocess.Popen(["python", script_name])
            self.log_message(f"Launched {script_name} successfully")
            
        except Exception as e:
            self.log_message(f"Error launching {script_name}: {str(e)}")
            




            
    def upload_video(self):
        """Handle video upload"""
        self.video_uploader.open_file_dialog() 



    def open_video_upload_window(self):
        """Open the upload video window in a new Tkinter Toplevel"""
        upload_window = tk.Toplevel(self.root)
        video_uploader = VideoUploader(upload_window)
        


if __name__ == "__main__":
    root = tk.Tk()
    app = GenderAgeDetectionApp(root)
    
    # Launch control panel if needed
    # from control_panel import launch_control_panel
    # launch_control_panel(app)
    
    root.mainloop()
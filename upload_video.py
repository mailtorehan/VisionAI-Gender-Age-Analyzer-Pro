import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
import requests
import os
from urllib.request import urlretrieve
import os
import threading
from datetime import datetime
from PIL import Image, ImageTk
from keras.models import load_model

class VideoUploader:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Analysis")
        self.root.geometry("800x600")
      
        # Video processing
        self.video_path = None
        self.capture = None
        self.is_processing = False
        self.current_frame = None
        self.paused = False
        self.processed_frames = 0
        
        # Detection models
        self.face_net = None
        self.gender_net = None
        self.age_net = None
        self.face_cascade = None
        self.gender_keras_model = None
        
        # Statistics
        self.male_count = 0
        self.female_count = 0
        self.age_distribution = {}
        self.total_frames = 0
        self.detection_history = []
        
        # Age and gender labels
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                        '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        
        # UI Setup
        self.setup_ui()
        
        # Load models
        self.load_models()

    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video display
        self.canvas = tk.Canvas(self.main_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Upload button
        self.upload_button = ttk.Button(
            control_frame,
            text="Upload Video",
            command=self.open_file_dialog
        )
        self.upload_button.pack(side=tk.LEFT, padx=5)
        
        # Play/Pause buttons
        self.play_button = ttk.Button(
            control_frame,
            text="Play",
            command=self.play_video,
            state=tk.DISABLED
        )
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(
            control_frame,
            text="Pause",
            command=self.pause_video,
            state=tk.DISABLED
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(self.main_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Stats labels
        self.male_label = ttk.Label(stats_frame, text="Male: 0")
        self.male_label.grid(row=0, column=0, padx=5)
        
        self.female_label = ttk.Label(stats_frame, text="Female: 0")
        self.female_label.grid(row=0, column=1, padx=5)
        
        self.age_label = ttk.Label(stats_frame, text="Age Distribution: ")
        self.age_label.grid(row=1, column=0, columnspan=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to upload video")
        ttk.Label(self.main_frame, textvariable=self.status_var).pack(side=tk.BOTTOM, fill=tk.X)

    def load_models(self):
        """Load all required models with comprehensive error handling and automatic downloads"""
        try:
            # Create models directory if it doesn't exist
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            
            # Load Haar Cascade (built into OpenCV)
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.face_cascade.empty():
                    raise Exception("Failed to load Haar Cascade")
            except Exception as e:
                self.status_var.set("Haar Cascade load failed")
                messagebox.showwarning("Warning", "Face detection accuracy may be reduced")
                self.face_cascade = None

            # Define DNN model files
            dnn_files = {
                'pb': {
                    'url': 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/opencv_face_detector_uint8.pb',
                    'local': os.path.join(model_dir, 'opencv_face_detector_uint8.pb')
                },
                'pbtxt': {
                    'url': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
                    'local': os.path.join(model_dir, 'opencv_face_detector.pbtxt')
                }
            }

            # Download missing DNN model files
            for file_type, file_info in dnn_files.items():
                if not os.path.exists(file_info['local']):
                    try:
                        self.status_var.set(f"Downloading {file_type} file...")
                        response = requests.get(file_info['url'], stream=True)
                        with open(file_info['local'], 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    except Exception as e:
                        self.status_var.set(f"Failed to download {file_type} file")
                        messagebox.showerror("Error", f"Couldn't download required {file_type} file\n{e}")
                        return

            # Load DNN face detector
            try:
                self.face_net = cv2.dnn.readNet(dnn_files['pb']['local'], 
                                            dnn_files['pbtxt']['local'])
                self.status_var.set("Loaded DNN face detector")
            except Exception as e:
                self.status_var.set("DNN face detector load failed")
                messagebox.showerror("Error", f"Failed to load DNN face detector\n{e}")
                self.face_net = None

            # Define Caffe model files
            caffe_models = {
                'gender': {
                    'prototxt': 'gender_deploy.prototxt',
                    'model': 'gender_net.caffemodel',
                    'url': 'https://drive.google.com/uc?export=download&id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ'
                },
                'age': {
                    'prototxt': 'age_deploy.prototxt',
                    'model': 'age_net.caffemodel',
                    'url': 'https://drive.google.com/uc?export=download&id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW'
                }
            }

            # Try loading Caffe models
            self.gender_net = None
            self.age_net = None
            
            for model_type, model_info in caffe_models.items():
                prototxt_path = os.path.join(model_dir, model_info['prototxt'])
                model_path = os.path.join(model_dir, model_info['model'])
                
                # Download if missing
                if not os.path.exists(model_path):
                    try:
                        self.status_var.set(f"Downloading {model_type} model...")
                        response = requests.get(model_info['url'], stream=True)
                        with open(model_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    except Exception as e:
                        self.status_var.set(f"Failed to download {model_type} model")
                        continue

                # Load model
                try:
                    if model_type == 'gender':
                        self.gender_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                    else:
                        self.age_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                    self.status_var.set(f"Loaded {model_type} model")
                except Exception as e:
                    self.status_var.set(f"Failed to load {model_type} model")
                    messagebox.showwarning("Warning", f"{model_type.capitalize()} detection unavailable")

            # Fallback to Keras model if Caffe models failed
            if self.gender_net is None:
                keras_model_path = os.path.join(model_dir, 'custom_gender_model.h5')
                if os.path.exists(keras_model_path):
                    try:
                        self.gender_keras_model = load_model(keras_model_path)
                        self.status_var.set("Loaded Keras gender model")
                    except Exception as e:
                        self.status_var.set("Failed to load Keras gender model")
                        messagebox.showerror("Error", "All gender models failed to load")
                else:
                    self.status_var.set("No gender models available")
                    messagebox.showerror("Error", "No gender detection models found")

        except Exception as e:
            self.status_var.set(f"Critical model loading error: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize models: {str(e)}")
            raise


    def open_file_dialog(self):
        """Open file dialog to select video"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            self.play_button.config(state=tk.NORMAL)
            self.prepare_video()

    def prepare_video(self):
        """Prepare video for processing"""
        try:
            if self.capture:
                self.capture.release()
                
            self.capture = cv2.VideoCapture(self.video_path)
            if not self.capture.isOpened():
                raise IOError("Cannot open video file")
                
            self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.processed_frames = 0
            self.progress_var.set(0)
            
            # Get first frame to initialize display
            ret, frame = self.capture.read()
            if ret:
                self.process_frame(frame, 0)
                
            self.status_var.set(f"Ready to process {self.total_frames} frames")
            
        except Exception as e:
            self.status_var.set(f"Error preparing video: {str(e)}")
            messagebox.showerror("Error", f"Video preparation failed: {str(e)}")

    def play_video(self):
        """Start video processing"""
        if self.is_processing and self.paused:
            self.paused = False
            self.process_video()
            return
            
        if not self.video_path or not self.capture:
            return
            
        # Reset video to beginning
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Reset counters
        self.male_count = 0
        self.female_count = 0
        self.age_distribution = {}
        self.detection_history = []
        self.processed_frames = 0
        self.progress_var.set(0)
        
        # Update UI
        self.male_label.config(text="Male: 0")
        self.female_label.config(text="Female: 0")
        self.age_label.config(text="Age Distribution: ")
        
        self.is_processing = True
        self.paused = False
        self.upload_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_video, daemon=True).start()

    def pause_video(self):
        """Pause video processing"""
        self.paused = True
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.status_var.set("Video paused")

    def process_video(self):
        """Process video frames with proper timing"""
        frame_skip = 3  # Process every 3rd frame for better performance
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30  # Default to ~30fps if unknown
        
        while self.is_processing and not self.paused and self.capture.isOpened():
            start_time = time.time()
            
            ret, frame = self.capture.read()
            if not ret:
                break

            self.processed_frames += 1

            # Skip frames for faster processing
            if self.processed_frames % frame_skip != 0:
                continue

            # Process the current frame
            self.process_frame(frame, self.processed_frames)

            # Update progress
            progress = (self.processed_frames / self.total_frames) * 100
            self.progress_var.set(progress)
            self.status_var.set(
                f"Processing: {self.processed_frames}/{self.total_frames} frames ({progress:.1f}%)"
            )

            # Calculate processing time and adjust delay
            processing_time = time.time() - start_time
            remaining_delay = max(1, delay - int(processing_time * 1000))
            
            # Update display and maintain timing
            self.root.update()
            time.sleep(remaining_delay / 1000)

        # Video finished or paused
        if not self.paused:
            self.is_processing = False
            self.play_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.upload_button.config(state=tk.NORMAL)
            self.status_var.set("Analysis complete")

    def process_frame(self, frame, frame_num):
        """Process each frame for detection"""
        self.current_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Process each face
        for (x1, y1, x2, y2) in faces:
            face_img = frame[y1:y2, x1:x2]
            
            # Predict gender and age
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
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{gender} {confidence:.1f}% - {age}"
            cv2.putText(frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add to history
            self.detection_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'gender': gender,
                'confidence': confidence,
                'age': age,
                'frame': frame_num
            })
        
        # Update display
        self.update_display(frame)
        
        # Update stats
        self.update_stats()

    def detect_faces(self, frame):
        """Detect faces using both cascade and DNN"""
        faces = []
        
        # Try DNN first
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), 
            [104, 117, 123], swapRB=False, crop=False
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([
                    frame.shape[1], frame.shape[0], 
                    frame.shape[1], frame.shape[0]
                ])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2, y2))
        
        # Fallback to Haar Cascade if no faces detected
        if not faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, 
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in cascade_faces:
                faces.append((x, y, x+w, y+h))
                
        return faces

    def predict_gender_age(self, face_img):
        """Predict gender and age using available models"""
        try:
            if face_img.size == 0:
                return "Unknown", 0.0, "Unknown"
                
            # Predict gender
            gender = "Unknown"
            confidence = 0.0
            
            if self.gender_net:  # Use Caffe model if available
                gender_blob = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                self.gender_net.setInput(gender_blob)
                gender_preds = self.gender_net.forward()
                gender = self.gender_list[gender_preds[0].argmax()]
                confidence = gender_preds[0].max() * 100
            elif self.gender_keras_model:  # Fall back to Keras
                img = cv2.resize(face_img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)
                pred = self.gender_keras_model.predict(img)[0][0]
                gender = "Male" if pred > 0.5 else "Female"
                confidence = abs(pred - 0.5) * 200  # Convert to 0-100% scale
            
            # Predict age (only using Caffe model)
            age = "Unknown"
            if self.age_net:
                age_blob = cv2.dnn.blobFromImage(
                    face_img, 1.0, (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                self.age_net.setInput(age_blob)
                age_preds = self.age_net.forward()
                age = self.age_list[age_preds[0].argmax()]
            
            return gender, confidence, age
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0.0, "Unknown"

    def update_display(self, frame):
        """Update the displayed image"""
        try:
            # Convert to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img = Image.fromarray(rgb_image)
            
            # Resize to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 0 and canvas_height > 0:
                img_ratio = img.width / img.height
                canvas_ratio = canvas_width / canvas_height
                
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
            print(f"Display update error: {e}")

    def update_stats(self):
        """Update statistics display"""
        self.male_label.config(text=f"Male: {self.male_count}")
        self.female_label.config(text=f"Female: {self.female_count}")
        
        # Format age distribution
        age_text = "Age Distribution: "
        for age, count in self.age_distribution.items():
            age_text += f"{age}:{count} "
        self.age_label.config(text=age_text)

    def on_closing(self):
        """Clean up when window closes"""
        self.is_processing = False
        if self.capture:
            self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoUploader(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
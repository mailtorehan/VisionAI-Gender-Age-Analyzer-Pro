import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import time
import os
import threading
from datetime import datetime
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Detection")
        self.root.geometry("800x600")
        
        # Camera setup
        self.camera_index = 0
        self.capture = None
        self.is_capturing = False
        self.current_frame = None
        
        # Detection models
        self.face_net = None
        self.gender_net = None
        self.age_net = None
        self.face_cascade = None
        
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
        
        # Start camera
        self.start_capture()

    def setup_ui(self):
        """Set up the user interface"""
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera display
        self.canvas = tk.Canvas(self.main_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Start/Stop buttons
        self.start_button = ttk.Button(
            control_frame,
            text="Start Camera",
            command=self.start_capture
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Camera",
            command=self.stop_capture,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(self.main_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Stats labels
        ttk.Label(stats_frame, text=f"Male: {self.male_count}").grid(row=0, column=0, padx=5)
        ttk.Label(stats_frame, text=f"Female: {self.female_count}").grid(row=0, column=1, padx=5)
        ttk.Label(stats_frame, text="Age Distribution:").grid(row=1, column=0, columnspan=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(self.main_frame, textvariable=self.status_var).pack(side=tk.BOTTOM, fill=tk.X)

    def load_models(self):
        """Load all required models"""
        try:
            # Load face detection models
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Load DNN models
            face_proto = "models/opencv_face_detector.pbtxt"
            face_model = "models/opencv_face_detector_uint8.pb"
            self.face_net = cv2.dnn.readNet(face_model, face_proto)
            
            # Load gender model
            gender_proto = "models/gender_deploy.prototxt"
            gender_model = "models/gender_net.caffemodel"
            self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
            
            # Load age model
            age_proto = "models/age_deploy.prototxt"
            age_model = "models/age_net.caffemodel"
            self.age_net = cv2.dnn.readNet(age_model, age_proto)
            
            self.status_var.set("Models loaded successfully")
        except Exception as e:
            self.status_var.set(f"Error loading models: {str(e)}")
            print(f"Error loading models: {e}")

    def start_capture(self):
        """Start camera capture"""
        if self.is_capturing:
            return
            
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                raise IOError("Cannot open camera")
                
            self.is_capturing = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Camera started")
            
            # Start capturing frames
            self.capture_frames()
        except Exception as e:
            self.status_var.set(f"Error starting camera: {str(e)}")

    def stop_capture(self):
        """Stop camera capture"""
        self.is_capturing = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Camera stopped")

    def capture_frames(self):
        """Capture and process frames"""
        if not self.is_capturing:
            return
            
        ret, frame = self.capture.read()
        if ret:
            # Process frame
            self.process_frame(frame)
            
            # Schedule next frame
            self.root.after(10, self.capture_frames)
        else:
            self.status_var.set("Error reading frame")
            self.stop_capture()

    def process_frame(self, frame):
        """Process each frame for detection"""
        self.total_frames += 1
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
                'age': age
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
                box = detections[0, 0, i, 3:7] * np.array(
                    frame.shape[1], frame.shape[0], 
                    frame.shape[1], frame.shape[0]
                )
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
        """Predict gender and age for a face"""
        try:
            if face_img.size == 0:
                return "Unknown", 0.0, "Unknown"
                
            # Preprocess for gender detection
            gender_blob = cv2.dnn.blobFromImage(
                face_img, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            self.gender_net.setInput(gender_blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            gender_confidence = gender_preds[0].max() * 100
            
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
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0.0, "Unknown"

    def update_display(self, frame):
        """Update the displayed image"""
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

    def update_stats(self):
        """Update statistics display"""
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame) and widget.winfo_name() == "!labelframe":
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label):
                        if "Male:" in child.cget("text"):
                            child.config(text=f"Male: {self.male_count}")
                        elif "Female:" in child.cget("text"):
                            child.config(text=f"Female: {self.female_count}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
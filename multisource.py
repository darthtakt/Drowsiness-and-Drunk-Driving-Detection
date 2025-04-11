from flask import Flask, request, jsonify, render_template_string
import base64
import cv2
import numpy as np
import time
from threading import Lock
from queue import Queue
import concurrent.futures
import dlib
from scipy.spatial import distance as dist
import os
import pygame
import threading
from typing import Dict, Tuple, Optional
import json
from datetime import datetime, timedelta
import sqlite3
import os.path
from collections import defaultdict
# Fix matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import random  # Add this for random sampling of EAR values

# Configuration - Optimized for multiple streams
MAX_WORKERS = 4  # Reduced for better performance
FRAME_QUEUE_SIZE = 30  # Reduced for lower latency
MAX_FRAME_RATE = 30  # Optimal FPS
FRAME_SKIP_THRESHOLD = 0.8  # More aggressive frame skipping
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
SAVE_FRAMES = False  # Set to True for debugging

# Drowsiness detection constants
EAR_THRESHOLD = 0.24  # Adjusted for general use
CONSEC_FRAMES = 15    # Number of consecutive frames to trigger drowsiness alert
ALERT_SOUND_PATH = "alert_sound.wav"  # Update with your actual path

# Alcohol detection constants
ALCOHOL_DETECTION_ENABLED = True
ALCOHOL_ALERT_SOUND_PATH = "alcohol_alert.wav"  # Path to alcohol alert sound
ALCOHOL_REDNESS_THRESHOLD = 0.1  # Reduced from 0.5 to be much more sensitive
ALCOHOL_CONSEC_FRAMES = 10  # Number of consecutive frames to confirm alcohol detection

# Debugging constants
DEBUG_ALCOHOL_DETECTION = True  # Set to False in production

# Analytics configuration
ENABLE_ANALYTICS = True
DB_PATH = "driver_data.db"
ANALYTICS_RETENTION_DAYS = 30  # How many days to keep data

# Global state for drowsiness detection
drowsy_counter = {}  # Track consecutive frames for each driver
sound_playing = {}   # Track if sound is playing for each driver
active_drivers = {}  # Track active drivers and their last activity time
DRIVER_TIMEOUT = 30  # Seconds after which a driver is considered inactive

# Global state for alcohol detection
alcohol_counter = {}  # Track consecutive frames for alcohol detection
alcohol_detected = {}  # Track if alcohol was detected for each driver
alcohol_sound_playing = {}  # Track if alcohol alert is playing for each driver

# Initialize pygame for audio playback with more channels
pygame.mixer.init()
pygame.mixer.set_num_channels(16)  # Increase from default 8 to 16 channels

# Generate default alert sounds if they don't exist
def generate_default_alert_sounds():
    import wave
    import struct
    
    def create_beep_wav(filename, frequency=440, duration=1.0, volume=0.8, repeat=3):
        # Create a simple beeping sound
        rate = 44100  # samples per second
        total_frames = int(rate * duration)
        
        # Open file for writing
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 2 bytes = 16 bits
            wf.setframerate(rate)
            
            # Generate samples
            for _ in range(repeat):
                for i in range(total_frames):
                    t = float(i) / rate  # time in seconds
                    # Generate sine wave with envelope
                    env = 1.0 if t < duration * 0.8 else 1.0 - ((t - duration * 0.8) / (duration * 0.2))
                    value = int(volume * 32767.0 * env * math.sin(2 * math.pi * frequency * t))
                    data = struct.pack('<h', value)
                    wf.writeframes(data)
                
                # Add small pause between beeps
                silence = struct.pack('<h', 0) * int(rate * 0.2)
                wf.writeframes(silence)
        
        print(f"Created default alert sound: {filename}")
    
    # Generate drowsiness alert (higher frequency, more urgent)
    if not os.path.exists(ALERT_SOUND_PATH):
        create_beep_wav(ALERT_SOUND_PATH, frequency=880, duration=0.5, repeat=5)
    
    # Generate alcohol alert with MAXIMUM VOLUME (1.2 will clip but create louder perceived sound)
    if not os.path.exists(ALCOHOL_ALERT_SOUND_PATH):
        # Use lower frequency (more alarming) with higher volume and more repeats
        create_beep_wav(ALCOHOL_ALERT_SOUND_PATH, frequency=330, duration=0.7, repeat=6, volume=1.2)

# Add math module for sine wave generation
import math

# Call this after pygame initialization
generate_default_alert_sounds()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS drowsiness_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        driver_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        event_type TEXT NOT NULL,
        ear_value REAL,
        session_id TEXT NOT NULL
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS driver_sessions (
        session_id TEXT PRIMARY KEY,
        driver_id TEXT NOT NULL,
        start_time DATETIME NOT NULL,
        end_time DATETIME,
        frames_processed INTEGER DEFAULT 0,
        drowsiness_detected_count INTEGER DEFAULT 0
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS alcohol_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        driver_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        event_type TEXT NOT NULL,
        redness_value REAL,
        session_id TEXT NOT NULL
    )
    ''')
    
    # Add alcohol_detected_count column if it doesn't exist
    try:
        c.execute("SELECT alcohol_detected_count FROM driver_sessions LIMIT 1")
    except sqlite3.OperationalError:
        print("[INFO] Adding alcohol_detected_count column to driver_sessions table")
        c.execute("ALTER TABLE driver_sessions ADD COLUMN alcohol_detected_count INTEGER DEFAULT 0")
    
    conn.commit()
    conn.close()

def cleanup_old_data():
    cutoff_date = datetime.now() - timedelta(days=ANALYTICS_RETENTION_DAYS)
    conn = get_db_connection()
    conn.execute("DELETE FROM drowsiness_events WHERE timestamp < ?", (cutoff_date,))
    conn.execute("DELETE FROM driver_sessions WHERE start_time < ?", (cutoff_date,))
    conn.commit()
    conn.close()

class SessionManager:
    def __init__(self):
        self.active_sessions = {}  # {driver_id: {session_id, start_time, frame_count}}
        
    def start_session(self, driver_id):
        if driver_id not in self.active_sessions:
            session_id = f"{driver_id}_{int(time.time())}"
            self.active_sessions[driver_id] = {
                "session_id": session_id,
                "start_time": datetime.now(),
                "frame_count": 0,
                "drowsiness_count": 0
            }
            
            conn = get_db_connection()
            conn.execute(
                "INSERT INTO driver_sessions (session_id, driver_id, start_time) VALUES (?, ?, ?)",
                (session_id, driver_id, datetime.now())
            )
            conn.commit()
            conn.close()
            
        return self.active_sessions[driver_id]["session_id"]
    
    def update_session(self, driver_id, drowsiness_detected=False, alcohol_detected=False):
        if driver_id in self.active_sessions:
            self.active_sessions[driver_id]["frame_count"] += 1
            if drowsiness_detected:
                self.active_sessions[driver_id]["drowsiness_count"] += 1
            
            alcohol_detected_count = 1 if alcohol_detected else 0
                    
            conn = get_db_connection()
            conn.execute(
                "UPDATE driver_sessions SET frames_processed = frames_processed + 1, "
                "drowsiness_detected_count = drowsiness_detected_count + ?, "
                "alcohol_detected_count = alcohol_detected_count + ? "
                "WHERE session_id = ?",
                (1 if drowsiness_detected else 0, alcohol_detected_count, self.active_sessions[driver_id]["session_id"])
            )
            conn.commit()
            conn.close()
    
    def end_session(self, driver_id):
        if driver_id in self.active_sessions:
            conn = get_db_connection()
            conn.execute(
                "UPDATE driver_sessions SET end_time = ? WHERE session_id = ?",
                (datetime.now(), self.active_sessions[driver_id]["session_id"])
            )
            conn.commit()
            conn.close()
            del self.active_sessions[driver_id]

    def cleanup_inactive_sessions(self):
        current_time = time.time()
        for driver_id in list(self.active_sessions.keys()):
            if driver_id not in active_drivers or (current_time - active_drivers[driver_id]) > DRIVER_TIMEOUT:
                self.end_session(driver_id)

session_manager = SessionManager()

def record_drowsiness_event(driver_id, event_type, ear_value=None):
    if not ENABLE_ANALYTICS:
        return
        
    session_id = session_manager.start_session(driver_id)
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO drowsiness_events (driver_id, timestamp, event_type, ear_value, session_id) VALUES (?, ?, ?, ?, ?)",
        (driver_id, datetime.now(), event_type, ear_value, session_id)
    )
    conn.commit()
    conn.close()

def record_alcohol_event(driver_id, event_type, redness_value=None):
    if not ENABLE_ANALYTICS:
        return
        
    session_id = session_manager.start_session(driver_id)
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO alcohol_events (driver_id, timestamp, event_type, redness_value, session_id) VALUES (?, ?, ?, ?, ?)",
        (driver_id, datetime.now(), event_type, redness_value, session_id)
    )
    conn.commit()
    conn.close()

def play_alert_sound(driver_id):
    """Play alert sound when drowsiness is detected"""
    if driver_id not in sound_playing or not sound_playing[driver_id]:
        try:
            # Check if file exists first
            if not os.path.exists(ALERT_SOUND_PATH):
                print(f"Warning: Alert sound file '{ALERT_SOUND_PATH}' not found. Using visual alerts only.")
                sound_playing[driver_id] = True
                return
                
            channel = pygame.mixer.Channel(hash(driver_id) % 8)  # Use driver_id to select channel (max 8 channels)
            sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
            channel.play(sound, loops=-1)  # Loop sound until stopped
            sound_playing[driver_id] = True
        except Exception as e:
            print(f"Error playing alert sound: {str(e)}")
            sound_playing[driver_id] = True  # Mark as playing to prevent repeated errors

def stop_alert_sound(driver_id):
    """Stop playing alert sound when driver is no longer drowsy"""
    if driver_id in sound_playing and sound_playing[driver_id]:
        try:
            channel = pygame.mixer.Channel(hash(driver_id) % 8)
            channel.stop()
            sound_playing[driver_id] = False
        except Exception as e:
            print(f"Error stopping alert sound: {str(e)}")

def play_alcohol_alert(driver_id):
    """Play alert sound when alcohol is detected"""
    global alcohol_sound_playing
    
    # Only play if not already playing for this driver
    if driver_id in alcohol_sound_playing and alcohol_sound_playing[driver_id]:
        print(f"Alcohol alert already playing for driver {driver_id}")
        return
        
    try:
        # Check if file exists first
        if not os.path.exists(ALCOHOL_ALERT_SOUND_PATH):
            print(f"Warning: Alcohol alert sound file '{ALCOHOL_ALERT_SOUND_PATH}' not found. Using visual alerts only.")
            return
        
        # Generate sound file if needed
        if not os.path.exists(ALCOHOL_ALERT_SOUND_PATH):
            # Create a louder, more attention-grabbing sound
            rate = 44100
            duration = 3.0
            # ... rest of sound generation code ...
            
        # Use two channels for amplified effect
        channel_indices = [
            min(hash(driver_id) % 8 + 8, pygame.mixer.get_num_channels() - 1),
            min(hash(driver_id + "extra") % 8 + 8, pygame.mixer.get_num_channels() - 2)
        ]
        
        sound = pygame.mixer.Sound(ALCOHOL_ALERT_SOUND_PATH)
        sound.set_volume(1.0)
        
        for channel_index in channel_indices:
            channel = pygame.mixer.Channel(channel_index)
            channel.set_volume(1.0)
            if not channel.get_busy():
                channel.play(sound, loops=-1)
                
        # Mark as playing
        alcohol_sound_playing[driver_id] = True
        print(f"Started alcohol alert for driver {driver_id}")
    except Exception as e:
        print(f"Error playing alcohol alert sound: {str(e)}")

def stop_alcohol_alert(driver_id):
    """Stop playing alcohol alert sound"""
    global alcohol_sound_playing
    
    try:
        # Use the same channel indices as in play_alcohol_alert
        channel_indices = [
            min(hash(driver_id) % 8 + 8, pygame.mixer.get_num_channels() - 1),
            min(hash(driver_id + "extra") % 8 + 8, pygame.mixer.get_num_channels() - 2)
        ]
        
        for channel_index in channel_indices:
            channel = pygame.mixer.Channel(channel_index)
            channel.stop()
        
        # Mark as not playing
        alcohol_sound_playing[driver_id] = False
        print(f"Stopped alcohol alert for driver {driver_id}")
    except Exception as e:
        print(f"Error stopping alcohol alert sound: {str(e)}")

def calculate_EAR(eye):
    """
    Calculate Eye Aspect Ratio (EAR) using the facial landmarks of one eye
    EAR = (A + B) / (2.0 * C)
    Where A is the distance between landmarks 1 and 5, 
          B is the distance between landmarks 2 and 4,
          C is the distance between landmarks 0 and 3
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_alcohol(face_frame, eyes_region):
    """
    Detect potential signs of alcohol consumption based on eye redness
    Returns: (is_detected, redness_value, eye_rects)
    """
    if face_frame is None or len(eyes_region) == 0:
        return False, 0.0, []
    
    # Extract eye regions from the frame with larger margins
    eye_rects = []
    for eye in [eyes_region[36:42], eyes_region[42:48]]:  # left and right eye
        x_min = int(np.min(eye[:, 0]))
        y_min = int(np.min(eye[:, 1]))
        x_max = int(np.max(eye[:, 0]))
        y_max = int(np.max(eye[:, 1]))
        # Add larger margin to capture more of the sclera
        margin = 15  # Increased from 10
        eye_rects.append((
            max(0, x_min - margin), 
            max(0, y_min - margin), 
            min(face_frame.shape[1], x_max + margin), 
            min(face_frame.shape[0], y_max + margin)
        ))
    
    # Measure redness in the eyes
    total_redness = 0.0
    for (x1, y1, x2, y2) in eye_rects:
        if x1 >= x2 or y1 >= y2:
            continue
            
        eye_roi = face_frame[y1:y2, x1:x2]
        if eye_roi.size == 0:
            continue
        
        # Enhance the red channel more aggressively
        b, g, r = cv2.split(eye_roi)
        enhanced_r = cv2.addWeighted(r, 2.0, r, 0, 0)  # Enhance red channel more (1.5 -> 2.0)
        enhanced_eye = cv2.merge([b, g, enhanced_r])
            
        # Convert to HSV color space for better color detection
        hsv_eye = cv2.cvtColor(enhanced_eye, cv2.COLOR_BGR2HSV)
        
        # More sensitive ranges for red color detection
        lower_red1 = np.array([0, 30, 50])     # Even more inclusive
        upper_red1 = np.array([20, 255, 255])  
        lower_red2 = np.array([150, 30, 50])   # Even more inclusive
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red detection
        mask1 = cv2.inRange(hsv_eye, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_eye, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Apply morphological operations to improve detection
        kernel = np.ones((5, 5), np.uint8)  # Larger kernel (3,3 -> 5,5)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        
        # Calculate redness percentage with higher weight
        redness_percentage = np.sum(red_mask > 0) / (red_mask.shape[0] * red_mask.shape[1])
        
        # Add additional detection for reddish pixels in RGB with more aggressive thresholds
        rg_ratio = np.mean(r) / (np.mean(g) + 0.01)  # Red to green ratio
        rb_ratio = np.mean(r) / (np.mean(b) + 0.01)  # Red to blue ratio
        
        # Combine HSV detection with RGB ratios (more weight on color ratios)
        color_bias = min(1.0, (rg_ratio - 0.7) + (rb_ratio - 0.7))  # Lower threshold (0.9 -> 0.7)
        if color_bias > 0:
            redness_percentage += color_bias * 0.3  # More boost (0.2 -> 0.3)
            
        total_redness += redness_percentage
    
    # Average redness across both eyes
    if len(eye_rects) > 0:
        avg_redness = total_redness / len(eye_rects)
    else:
        avg_redness = 0.0
    
    # Check if redness is above threshold (could lower this if needed)
    # ALCOHOL_REDNESS_THRESHOLD is already set to 0.1 which is quite sensitive
    is_alcohol_detected = avg_redness > ALCOHOL_REDNESS_THRESHOLD
    
    # Return eye_rects along with other values
    return is_alcohol_detected, avg_redness, eye_rects

def cleanup_inactive_drivers():
    current_time = time.time()
    inactive_drivers = []
    
    for driver_id, last_active in list(active_drivers.items()):
        if current_time - last_active > DRIVER_TIMEOUT:
            inactive_drivers.append(driver_id)
    
    for driver_id in inactive_drivers:
        if driver_id in drowsy_counter:
            del drowsy_counter[driver_id]
        if driver_id in sound_playing:
            stop_alert_sound(driver_id)
            del sound_playing[driver_id]
        if driver_id in active_drivers:
            del active_drivers[driver_id]
        
        session_manager.end_session(driver_id)
        print(f"[INFO] Driver {driver_id} marked as inactive and removed from tracking")

class FrameProcessor:
    def __init__(self):
        self.frame_queues = {}
        self.last_frame_times = {}
        self.frame_counters = {}
        self.lock = Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        print("[INFO] Loading facial landmark predictor...")
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            print(f"[ERROR] Shape predictor file not found: {model_path}")
            print("[INFO] You can download it from: https://github.com/davisking/dlib-models")
            raise FileNotFoundError(f"Shape predictor file not found: {model_path}")
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        
        self.lStart, self.lEnd = 42, 48
        self.rStart, self.rEnd = 36, 42
        
        self.cleanup_thread = threading.Thread(target=self._run_cleanup, daemon=True)
        self.cleanup_thread.start()

    def _run_cleanup(self):
        while True:
            cleanup_inactive_drivers()
            time.sleep(10)

    def process_frame(self, driver_id: str, frame_data: str) -> Tuple[Dict, int]:
        current_time = time.time()
        
        with self.lock:
            if driver_id not in self.frame_queues:
                self.frame_queues[driver_id] = Queue(maxsize=FRAME_QUEUE_SIZE)
                self.frame_counters[driver_id] = 0
                drowsy_counter[driver_id] = 0
                sound_playing[driver_id] = False
            
            active_drivers[driver_id] = current_time
            self.frame_counters[driver_id] = (self.frame_counters[driver_id] + 1) % PROCESS_EVERY_N_FRAMES
            
            queue = self.frame_queues[driver_id]
            if queue.qsize() > FRAME_QUEUE_SIZE * FRAME_SKIP_THRESHOLD or self.frame_counters[driver_id] != 0:
                return {"status": "skipped_for_performance"}, 202

            self.last_frame_times[driver_id] = current_time

        future = self.executor.submit(self._decode_and_process, driver_id, frame_data)
        return {"status": "processing"}, 202

    def _decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        try:
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_REDUCED_COLOR_2)
            if frame is None:
                return None
            return cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Frame decoding error: {str(e)}")
            return None

    def _decode_and_process(self, driver_id: str, frame_data: str) -> None:
        try:
            img = self._decode_frame(frame_data)
            if img is None:
                return

            processed, drowsiness_detected, alcohol_detected = self._custom_processing(driver_id, img)
            
            # Update to store frame data per driver
            global latest_frames
            _, buffer = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            latest_frames[driver_id] = {
                "frame": base64.b64encode(buffer).decode('utf-8'),
                "drowsiness_detected": drowsiness_detected,
                "alcohol_detected": alcohol_detected,
                "timestamp": time.time()
            }
            
            if __name__ == "__main__" and SAVE_FRAMES:
                self._save_frame(driver_id, processed)

        except Exception as e:
            print(f"Error processing frame from {driver_id}: {str(e)}")

    def _custom_processing(self, driver_id: str, frame: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector(gray, 0)
        
        session_manager.start_session(driver_id)
        drowsiness_detected = False
        is_alcohol_detected = False  # Renamed local variable to avoid conflict
        
        # Initialize alcohol counter if not present
        if driver_id not in alcohol_counter:
            alcohol_counter[driver_id] = 0
        if driver_id not in alcohol_detected:
            alcohol_detected[driver_id] = False
        
        for face in faces:
            shape = self.predictor(gray, face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            
            leftEye = shape_np[self.lStart:self.lEnd]
            rightEye = shape_np[self.rStart:self.rEnd]
            
            leftEAR = calculate_EAR(leftEye)
            rightEAR = calculate_EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face region for alcohol detection
            face_roi = frame[max(0, y):min(frame.shape[0], y+h), max(0, x):min(frame.shape[1], x+w)]
            
            # Alcohol detection
            if ALCOHOL_DETECTION_ENABLED:
                is_alcohol, redness_value, eye_rects = detect_alcohol(face_roi, shape_np)
                
                if is_alcohol:
                    alcohol_counter[driver_id] += 1
                    if alcohol_counter[driver_id] >= ALCOHOL_CONSEC_FRAMES:
                        cv2.putText(display_frame, "ALCOHOL DETECTED!", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        
                        is_alcohol_detected = True  # Use renamed local variable
                        alcohol_detected[driver_id] = True  # This uses the global dictionary
                        record_alcohol_event(driver_id, "alcohol_detected", redness_value)
                        print(f"Alcohol detected for driver {driver_id} - triggering alert")
                        play_alcohol_alert(driver_id)
                else:
                    if alcohol_counter[driver_id] >= ALCOHOL_CONSEC_FRAMES:
                        record_alcohol_event(driver_id, "alcohol_cleared", redness_value)
                        if alcohol_detected[driver_id]:
                            stop_alcohol_alert(driver_id)
                            alcohol_detected[driver_id] = False
                    
                    alcohol_counter[driver_id] = 0
                
                # Display alcohol detection info
                cv2.putText(display_frame, f"Redness: {redness_value:.2f}", (480, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if DEBUG_ALCOHOL_DETECTION:
                    # Draw eye regions used for alcohol detection
                    for (x1, y1, x2, y2) in eye_rects:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
                    
                    # Add debug text showing the actual value
                    cv2.putText(display_frame, f"Redness: {redness_value:.4f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Process eyes for drowsiness detection
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(display_frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(display_frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Original drowsiness detection code
            if ear < EAR_THRESHOLD:
                drowsy_counter[driver_id] += 1
                if drowsy_counter[driver_id] >= CONSEC_FRAMES:
                    cv2.putText(display_frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    drowsiness_detected = True
                    record_drowsiness_event(driver_id, "drowsiness_detected", ear)
                    
                    if not sound_playing.get(driver_id, False):
                        play_alert_sound(driver_id)
            else:
                if drowsy_counter[driver_id] >= CONSEC_FRAMES:
                    record_drowsiness_event(driver_id, "drowsiness_recovered", ear)
                
                drowsy_counter[driver_id] = 0
                if sound_playing.get(driver_id, False):
                    stop_alert_sound(driver_id)
            
            cv2.putText(display_frame, f"EAR: {ear:.2f}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        session_manager.update_session(driver_id, drowsiness_detected, is_alcohol_detected)
        return display_frame, drowsiness_detected, is_alcohol_detected

    def _save_frame(self, driver_id: str, frame: np.ndarray) -> None:
        timestamp = int(time.time() * 1000)
        os.makedirs(f"frames/{driver_id}", exist_ok=True)
        cv2.imwrite(f"frames/{driver_id}/{timestamp}.jpg", frame)

def generate_drowsiness_chart():
    conn = get_db_connection()
    data = conn.execute("""
        SELECT driver_id, COUNT(*) as event_count 
        FROM drowsiness_events 
        WHERE event_type = 'drowsiness_detected' 
        AND timestamp > ?
        GROUP BY driver_id
    """, (datetime.now() - timedelta(days=7),)).fetchall()
    conn.close()
    
    if not data:
        return None
        
    drivers = [row['driver_id'] for row in data]
    counts = [row['event_count'] for row in data]
    
    plt.figure(figsize=(10, 6))
    # Use red color for drowsiness
    plt.bar(drivers, counts, color='#e53935')
    plt.title('Drowsiness Events by Driver (Last 7 Days)')
    plt.xlabel('Driver ID')
    plt.ylabel('Number of Events')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    return img_data

def generate_alcohol_chart():
    conn = get_db_connection()
    data = conn.execute("""
        SELECT driver_id, COUNT(*) as event_count 
        FROM alcohol_events 
        WHERE event_type = 'alcohol_detected' 
        AND timestamp > ?
        GROUP BY driver_id
    """, (datetime.now() - timedelta(days=7),)).fetchall()
    conn.close()
    
    if not data:
        return None
        
    drivers = [row['driver_id'] for row in data]
    counts = [row['event_count'] for row in data]
    
    plt.figure(figsize=(10, 6))
    # Use blue color for alcohol
    plt.bar(drivers, counts, color='#4169E1')
    plt.title('Alcohol Detection Events by Driver (Last 7 Days)')
    plt.xlabel('Driver ID')
    plt.ylabel('Number of Events')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode('utf-8')
    plt.close()
    
    return img_data

app = Flask(__name__)
processor = FrameProcessor()

latest_frames = {}  # Dictionary to store latest frame for each driver

STREAM_VIEWER_HTML = """
<!DOCTYPE html>
<html data-theme="light">
<head>
    <title>Multi-Source Drowsiness Detection System</title>
    <style>
        :root {
            --bg-color: #f5f5f5;
            --text-color: #333;
            --card-bg: #ffffff;
            --card-shadow: 0 2px 4px rgba(0,0,0,0.1);
            --stat-bg: #e9e9e9;
            --header-color: #333;
            --border-color: #ddd;
            --link-color: #3498db;
            --normal-color: #4CAF50;
            --alert-color: #ff4d4d;
            --alcohol-color: #4169E1;
            --warning-color: #ff9933;
            --highlight-bg: #f0f0f0;
            --drowsy-color: #e53935;
            --chart-grid-color: #ddd;
        }
        
        [data-theme="dark"] {
            --bg-color: #1a1a1a;
            --text-color: #f0f0f0;
            --card-bg: #2d2d2d;
            --card-shadow: 0 2px 4px rgba(0,0,0,0.3);
            --stat-bg: #3d3d3d;
            --header-color: #e0e0e0;
            --border-color: #444;
            --link-color: #5dade2;
            --normal-color: #3d8b40;
            --alert-color: #e53935;
            --alcohol-color: #4e7fec;
            --warning-color: #f57c00;
            --highlight-bg: #333333;
            --drowsy-color: #ff5252;
            --chart-grid-color: #444;
        }
        
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: all 0.3s ease;
        }
        
        .container { max-width: 1300px; margin: 0 auto; }
        
        h1, h2, h3 { 
            color: var(--header-color); 
            text-align: center; 
        }
        
        .multi-stream-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            grid-template-rows: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .stream-panel {
            background-color: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            box-shadow: var(--card-shadow);
            display: flex;
            flex-direction: column;
            aspect-ratio: 4/3;
        }
        
        .stream-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 5px;
        }
        
        .driver-name {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .driver-status {
            display: flex;
            gap: 5px;
        }
        
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: var(--normal-color);
        }
        
        .status-indicator.alert {
            background-color: var(--alert-color);
            animation: pulse 1s infinite;
        }
        
        .status-indicator.warning {
            background-color: var(--warning-color);
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .driver-image {
            width: 100%;
            height: auto;
            aspect-ratio: 4/3;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid var(--border-color);
        }
        
        .driver-stats {
            margin-top: 10px;
            display: flex;
            justify-content: space-between;
        }
        
        .stat-bar-container {
            flex: 1;
            margin: 0 5px;
        }
        
        .stat-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 3px;
            font-size: 0.9em;
        }
        
        .stat-bar {
            height: 8px;
            border-radius: 4px;
            background-color: var(--stat-bg);
            overflow: hidden;
        }
        
        .stat-bar .fill {
            height: 100%;
            border-radius: 4px;
        }
        
        .drowsy-fill {
            background-color: var(--drowsy-color);
        }
        
        .alcohol-fill {
            background-color: var(--alcohol-color);
        }
        
        img { 
            max-width: 100%; 
            border-radius: 8px; 
        }
        
        .status-bar, .alcohol-status { 
            margin-top: 10px; 
            padding: 8px; 
            font-weight: bold; 
            text-align: center;
            border-radius: 5px;
            transition: background-color 0.5s ease;
            font-size: 0.9em;
        }
        
        .alert { background-color: var(--alert-color); color: white; }
        .warning { background-color: var(--warning-color); color: white; }
        .normal { background-color: var(--normal-color); color: white; }
        
        .driver-selector {
            text-align: center;
            margin: 20px 0;
        }
        
        select, button {
            padding: 8px;
            border-radius: 4px;
            font-size: 16px;
            background-color: var(--card-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
        }
        
        button:hover {
            cursor: pointer;
            opacity: 0.9;
        }
        
        .global-stats {
            display: flex;
            justify-content: space-around;
            margin: 15px 0;
            background-color: var(--card-bg);
            padding: 15px;
            border-radius: 8px;
            box-shadow: var(--card-shadow);
        }
        
        .stat-item {
            background-color: var(--stat-bg);
            padding: 8px 15px;
            border-radius: 4px;
        }
        
        .header-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .theme-toggle {
            padding: 8px 16px;
            background-color: var(--highlight-bg);
            border-radius: 20px;
            display: flex;
            align-items: center;
        }
        
        .theme-toggle i {
            margin-right: 5px;
        }
        
        .nav-links {
            display: flex;
            gap: 20px;
            justify-content: center;
        }
        
        .nav-link {
            text-decoration: none;
            color: var(--link-color);
            font-weight: bold;
        }
        
        .no-drivers-message {
            text-align: center;
            padding: 40px;
            background-color: var(--card-bg);
            border-radius: 8px;
            box-shadow: var(--card-shadow);
            margin: 20px 0;
        }
    </style>
    
    <!-- Add Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <!-- Add Chart.js for better charts -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header-controls">
            <div class="nav-links">
                <a href="/" class="nav-link">Live Monitoring</a>
                <a href="/analytics" class="nav-link">Analytics Dashboard</a>
            </div>
            <button id="themeToggle" class="theme-toggle">
                <i class="fas fa-moon"></i> Dark Mode
            </button>
        </div>
        
        <h1>Multi-Driver Monitoring System</h1>
        
        <div class="global-stats">
            <div class="stat-item" id="frameRate">Frame Rate: -- fps</div>
            <div class="stat-item" id="latency">Latency: -- ms</div>
            <div class="stat-item" id="activeDrivers">Active Drivers: 0</div>
            <div class="stat-item" id="totalAlerts">Total Alerts: 0</div>
        </div>
        
        <div id="noDriversMessage" class="no-drivers-message">
            <h3><i class="fas fa-car"></i> Waiting for drivers to connect...</h3>
            <p>No active drivers detected. Driver panels will appear automatically when connected.</p>
        </div>
        
        <div id="multiStreamContainer" class="multi-stream-container">
            <!-- Driver panels will be dynamically added here -->
        </div>
    </div>
    
    <script>
        // Theme switching functionality
        const themeToggle = document.getElementById('themeToggle');
        const htmlElement = document.documentElement;
        
        // Load saved theme preference or use default
        if (localStorage.getItem('theme') === 'dark') {
            htmlElement.setAttribute('data-theme', 'dark');
            themeToggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
        }
        
        themeToggle.addEventListener('click', () => {
            if (htmlElement.getAttribute('data-theme') === 'dark') {
                htmlElement.setAttribute('data-theme', 'light');
                themeToggle.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
                localStorage.setItem('theme', 'light');
            } else {
                htmlElement.setAttribute('data-theme', 'dark');
                themeToggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
                localStorage.setItem('theme', 'dark');
            }
        });
        
        // Driver monitoring functionality
        let frameCount = 0;
        let lastFrameTime = Date.now();
        let frameRates = [];
        let knownDrivers = new Set();
        let driverViolations = {};
        let driverFrameCounts = {};
        let driverChart = null;
        
        // Function to update all driver streams simultaneously
        function updateAllStreams() {
            const startTime = Date.now();
            
            fetch('/get_all_frames')
                .then(response => response.json())
                .then(data => {
                    const latency = Date.now() - startTime;
                    document.getElementById('latency').textContent = `Latency: ${latency} ms`;
                    
                    if (data.drivers && data.drivers.length > 0) {
                        document.getElementById('noDriversMessage').style.display = 'none';
                        document.getElementById('activeDrivers').textContent = `Active Drivers: ${data.drivers.length}`;
                        
                        const multiStreamContainer = document.getElementById('multiStreamContainer');
                        
                        // Limit to 4 streams maximum
                        const driversToDisplay = data.drivers.slice(0, 4);
                        
                        // Calculate which panels to keep/remove
                        const existingPanelIds = new Set(
                            Array.from(multiStreamContainer.children)
                            .map(panel => panel.getAttribute('data-driver-id'))
                        );
                        
                        const panelsToKeep = new Set(driversToDisplay.map(d => d.driver_id));
                        
                        // Remove panels that aren't in the first 4 anymore
                        for (const panelId of existingPanelIds) {
                            if (!panelsToKeep.has(panelId)) {
                                const panel = document.getElementById(`driver-panel-${panelId}`);
                                if (panel) panel.remove();
                            }
                        }
                        
                        // Count total alerts for the global counter
                        let totalAlerts = 0;
                        
                        // Process up to 4 drivers
                        driversToDisplay.forEach((driver, index) => {
                            const driverId = driver.driver_id;
                            
                            // Track driver frame counts for calculating percentages
                            if (!driverFrameCounts[driverId]) {
                                driverFrameCounts[driverId] = {
                                    total: 0,
                                    drowsy: 0,
                                    alcohol: 0
                                };
                            }
                            
                            driverFrameCounts[driverId].total++;
                            if (driver.drowsiness_detected) {
                                driverFrameCounts[driverId].drowsy++;
                            }
                            if (driver.alcohol_detected) {
                                driverFrameCounts[driverId].alcohol++;
                            }
                            
                            // Initialize driver in violations tracking if new
                            if (!driverViolations[driverId]) {
                                driverViolations[driverId] = {
                                    drowsiness: 0,
                                    alcohol: 0,
                                    lastActive: new Date(),
                                    status: 'Active'
                                };
                                knownDrivers.add(driverId);
                            } else {
                                driverViolations[driverId].lastActive = new Date();
                                driverViolations[driverId].status = 'Active';
                            }
                            
                            // Track violations
                            if (driver.drowsiness_detected) {
                                driverViolations[driverId].drowsiness++;
                                totalAlerts++;
                            }
                            
                            if (driver.alcohol_detected) {
                                driverViolations[driverId].alcohol++;
                                totalAlerts++;
                            }
                            
                            // Check if panel exists or create new one
                            let panel = document.getElementById(`driver-panel-${driverId}`);
                            if (!panel) {
                                panel = createDriverPanel(driverId);
                                
                                // Place panel in specific position based on index
                                if (index < 4) {
                                    multiStreamContainer.appendChild(panel);
                                }
                            }
                            
                            // Update panel content
                            const img = panel.querySelector('.driver-image');
                            img.src = 'data:image/jpeg;base64,' + driver.frame;
                            
                            // Update status indicators
                            const drowsyIndicator = panel.querySelector('.drowsy-indicator');
                            if (driver.drowsiness_detected) {
                                drowsyIndicator.classList.add('alert');
                            } else {
                                drowsyIndicator.classList.remove('alert');
                            }
                            
                            const alcoholIndicator = panel.querySelector('.alcohol-indicator');
                            if (driver.alcohol_detected) {
                                alcoholIndicator.classList.add('warning');
                            } else {
                                alcoholIndicator.classList.remove('warning');
                            }
                            
                            // Update performance bars
                            const drowsyPercent = ((driverFrameCounts[driverId].drowsy / driverFrameCounts[driverId].total) * 100).toFixed(1);
                            const alcoholPercent = ((driverFrameCounts[driverId].alcohol / driverFrameCounts[driverId].total) * 100).toFixed(1);
                            
                            panel.querySelector('.drowsy-fill').style.width = drowsyPercent + '%';
                            panel.querySelector('.alcohol-fill').style.width = alcoholPercent + '%';
                            panel.querySelector('.drowsy-percent').textContent = drowsyPercent + '%';
                            panel.querySelector('.alcohol-percent').textContent = alcoholPercent + '%';
                        });
                        
                        document.getElementById('totalAlerts').textContent = `Total Alerts: ${totalAlerts}`;
                        
                        // Make sure we only have 4 panels maximum
                        while (multiStreamContainer.children.length > 4) {
                            multiStreamContainer.removeChild(multiStreamContainer.lastChild);
                        }
                        
                        // Add empty panels if we have fewer than 4 drivers
                        while (multiStreamContainer.children.length < 4) {
                            const emptyPanel = document.createElement('div');
                            emptyPanel.className = 'stream-panel empty-panel';
                            emptyPanel.innerHTML = '<div class="empty-stream-message">No driver connected</div>';
                            multiStreamContainer.appendChild(emptyPanel);
                        }
                    } else {
                        document.getElementById('noDriversMessage').style.display = 'block';
                        document.getElementById('activeDrivers').textContent = 'Active Drivers: 0';
                        
                        // Fill with 4 empty panels
                        const multiStreamContainer = document.getElementById('multiStreamContainer');
                        multiStreamContainer.innerHTML = '';
                        
                        for (let i = 0; i < 4; i++) {
                            const emptyPanel = document.createElement('div');
                            emptyPanel.className = 'stream-panel empty-panel';
                            emptyPanel.innerHTML = '<div class="empty-stream-message">No driver connected</div>';
                            multiStreamContainer.appendChild(emptyPanel);
                        }
                    }
                    
                    frameCount++;
                    const now = Date.now();
                    if (now - lastFrameTime >= 1000) {
                        const fps = frameCount / ((now - lastFrameTime) / 1000);
                        frameRates.push(fps);
                        if (frameRates.length > 5) frameRates.shift();
                        
                        const avgFps = frameRates.reduce((a, b) => a + b, 0) / frameRates.length;
                        document.getElementById('frameRate').textContent = `Frame Rate: ${avgFps.toFixed(1)} fps`;
                        
                        frameCount = 0;
                        lastFrameTime = now;
                    }
                    
                    // Schedule next update
                    setTimeout(updateAllStreams, 50);
                })
                .catch(error => {
                    console.error('Error:', error);
                    setTimeout(updateAllStreams, 1000);
                });
        }

        function createDriverPanel(driverId) {
            const panel = document.createElement('div');
            panel.className = 'stream-panel';
            panel.id = `driver-panel-${driverId}`;
            panel.setAttribute('data-driver-id', driverId);
            
            panel.innerHTML = `
                <div class="stream-header">
                    <div class="driver-name">Driver: ${driverId}</div>
                    <div class="driver-status">
                        <span>Drowsy:</span>
                        <div class="status-indicator drowsy-indicator"></div>
                        <span>Alcohol:</span>
                        <div class="status-indicator alcohol-indicator"></div>
                    </div>
                </div>
                <img class="driver-image" src="" alt="Driver Stream">
                <div class="driver-stats">
                    <div class="stat-bar-container">
                        <div class="stat-label">
                            <span>Drowsiness</span>
                            <span class="drowsy-percent">0%</span>
                        </div>
                        <div class="stat-bar">
                            <div class="fill drowsy-fill" style="width: 0%"></div>
                        </div>
                    </div>
                    <div class="stat-bar-container">
                        <div class="stat-label">
                            <span>Alcohol</span>
                            <span class="alcohol-percent">0%</span>
                        </div>
                        <div class="stat-bar">
                            <div class="fill alcohol-fill" style="width: 0%"></div>
                        </div>
                    </div>
                </div>
            `;
            
            return panel;
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            updateAllStreams();
        });
    </script>
</body>
</html>
"""

ANALYTICS_HTML = """
<!DOCTYPE html>
<html data-theme="light">
<head>
    <title>Driver Drowsiness Analytics</title>
    <style>
        :root {
            --bg-color: #f5f5f5;
            --text-color: #333;
            --card-bg: #ffffff;
            --card-shadow: 0 2px 4px rgba(0,0,0,0.1);
            --stat-bg: #f0f0f0;
            --header-color: #333;
            --border-color: #ddd;
            --link-color: #3498db;
            --table-header-bg: #f2f2f2;
            --value-color: #2c3e50;
            --label-color: #7f8c8d;
        }
        
        [data-theme="dark"] {
            --bg-color: #1a1a1a;
            --text-color: #f0f0f0;
            --card-bg: #2d2d2d;
            --card-shadow: 0 2px 4px rgba(0,0,0,0.3);
            --stat-bg: #3d3d3d;
            --header-color: #e0e0e0;
            --border-color: #444;
            --link-color: #5dade2;
            --table-header-bg: #333333;
            --value-color: #ecf0f1;
            --label-color: #bdc3c7;
        }
        
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: all 0.3s ease;
        }
        
        .container { max-width: 1200px; margin: 0 auto; }
        
        h1, h2 { 
            color: var(--header-color); 
        }
        
        .header-controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .theme-toggle {
            padding: 8px 16px;
            background-color: var(--stat-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            color: var(--text-color);
            cursor: pointer;
            display: flex;
            align-items: center;
            font-size: 14px;
        }
        
        .theme-toggle i {
            margin-right: 5px;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 8px;
            box-shadow: var(--card-shadow);
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .stat-box {
            background-color: var(--stat-bg);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--value-color);
        }
        
        .stat-label {
            color: var(--label-color);
            margin-top: 5px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }
        
        th {
            background-color: var(--table-header-bg);
        }
        
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        
        .chart-image {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }
        
        .nav-links {
            display: flex;
            gap: 20px;
        }
        
        .nav-link {
            text-decoration: none;
            color: var(--link-color);
            font-weight: bold;
        }
        
        .empty-state {
            padding: 20px;
            text-align: center;
            color: var(--label-color);
            font-style: italic;
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
    </style>
    <!-- Add Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
    <!-- Add Chart.js for better charts -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header-controls">
            <div class="nav-links">
                <a href="/" class="nav-link">Live Monitoring</a>
                <a href="/analytics" class="nav-link">Analytics Dashboard</a>
            </div>
            <button id="themeToggle" class="theme-toggle">
                <i class="fas fa-moon"></i> Dark Mode
            </button>
        </div>
        
        <h1>Driver Drowsiness Analytics</h1>
        
        <div class="card">
            <h2>Summary Statistics</h2>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value">{{ total_sessions }}</div>
                    <div class="stat-label">Total Sessions</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ total_drowsiness_events }}</div>
                    <div class="stat-label">Drowsiness Events</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{{ total_alcohol_events }}</div>
                    <div class="stat-label">Alcohol Events</div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Driver Alert Events</h2>
            <div class="chart-grid">
                <div class="chart-container">
                    <h3>Drowsiness Events</h3>
                    {% if drowsiness_chart %}
                    <img class="chart-image" src="data:image/png;base64,{{ drowsiness_chart }}" alt="Drowsiness Chart">
                    {% else %}
                    <div class="empty-state">No data available for drowsiness events.</div>
                    {% endif %}
                </div>
                
                <div class="chart-container">
                    <h3>Alcohol Detection Events</h3>
                    {% if alcohol_chart %}
                    <img class="chart-image" src="data:image/png;base64,{{ alcohol_chart }}" alt="Alcohol Chart">
                    {% else %}
                    <div class="empty-state">No data available for alcohol detection events.</div>
                    {% endif %}
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>Driver Performance Metrics</h2>
            <div class="chart-container" style="height: 300px;">
                <canvas id="driverPerformanceChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Driver Violations Summary</h2>
            <table id="violationsTable">
                <thead>
                    <tr>
                        <th>Driver ID</th>
                        <th>Drowsiness Violations</th>
                        <th>Alcohol Violations</th>
                        <th>Status</th>
                        <th>Last Active</th>
                        <th>Drowsiness %</th>
                        <th>Alcohol %</th>
                    </tr>
                </thead>
                <tbody id="violationsTableBody">
                    <!-- Table will be populated by JavaScript -->
                </tbody>
            </table>
        </div>

        <script>
            const htmlElement = document.documentElement;
            const themeToggle = document.getElementById('themeToggle');
            
            // Theme switching functionality
            themeToggle.addEventListener('click', () => {
                if (htmlElement.getAttribute('data-theme') === 'dark') {
                    htmlElement.setAttribute('data-theme', 'light');
                    themeToggle.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
                    localStorage.setItem('theme', 'light');
                } else {
                    htmlElement.setAttribute('data-theme', 'dark');
                    themeToggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
                    localStorage.setItem('theme', 'dark');
                }
                
                // Update chart theme when toggling
                updateChartTheme();
            });
            
            function updateChartTheme() {
                if (driverChart) {
                    const isDarkMode = htmlElement.getAttribute('data-theme') === 'dark';
                    const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
                    const textColor = isDarkMode ? '#f0f0f0' : '#333';
                    
                    driverChart.options.scales.y.ticks.color = textColor;
                    driverChart.options.scales.y.grid.color = gridColor;
                    driverChart.options.scales.y.title.color = textColor;
                    driverChart.options.scales.x.ticks.color = textColor;
                    driverChart.options.scales.x.grid.color = gridColor;
                    driverChart.options.scales.x.title.color = textColor;
                    driverChart.options.plugins.legend.labels.color = textColor;
                    driverChart.options.plugins.title.color = textColor;
                    
                    driverChart.update();
                }
            }
            
            let driverChart = null;
            let driverViolations = {};
            let driverFrameCounts = {};
            
            function updatePerformanceData() {
                console.log("Fetching performance data...");
                fetch('/driver_performance_data')
                    .then(response => response.json())
                    .then(data => {
                        console.log("Received performance data:", data);
                        driverViolations = data.violations || {};
                        driverFrameCounts = data.frameCounts || {};
                        
                        // Update the UI with the new data
                        updatePerformanceChart();
                        updateViolationsTable();
                        
                        // Schedule next update
                        setTimeout(updatePerformanceData, 5000);
                    })
                    .catch(error => {
                        console.error('Error fetching performance data:', error);
                        setTimeout(updatePerformanceData, 10000);
                    });
            }
            
            function updatePerformanceChart() {
                const ctx = document.getElementById('driverPerformanceChart').getContext('2d');
                
                const drivers = Object.keys(driverFrameCounts);
                const drowsyData = drivers.map(driverId => {
                    if (!driverFrameCounts[driverId] || driverFrameCounts[driverId].total === 0) return 0;
                    return (driverFrameCounts[driverId].drowsy / driverFrameCounts[driverId].total * 100).toFixed(1);
                });
                
                const alcoholData = drivers.map(driverId => {
                    if (!driverFrameCounts[driverId] || driverFrameCounts[driverId].total === 0) return 0;
                    return (driverFrameCounts[driverId].alcohol / driverFrameCounts[driverId].total * 100).toFixed(1);
                });
                
                const isDarkMode = htmlElement.getAttribute('data-theme') === 'dark';
                const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
                const textColor = isDarkMode ? '#f0f0f0' : '#333';
                
                if (driverChart) {
                    driverChart.data.labels = drivers;
                    driverChart.data.datasets[0].data = drowsyData;
                    driverChart.data.datasets[1].data = alcoholData;
                    driverChart.update();
                } else {
                    driverChart = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: drivers,
                            datasets: [
                                {
                                    label: 'Drowsiness %',
                                    data: drowsyData,
                                    backgroundColor: 'rgba(229, 57, 53, 0.7)',
                                    borderColor: 'rgb(229, 57, 53)',
                                    borderWidth: 1
                                },
                                {
                                    label: 'Alcohol %',
                                    data: alcoholData,
                                    backgroundColor: 'rgba(65, 105, 225, 0.7)',
                                    borderColor: 'rgb(65, 105, 225)',
                                    borderWidth: 1
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100,
                                    ticks: {
                                        callback: value => value + '%',
                                        color: textColor
                                    },
                                    grid: {
                                        color: gridColor
                                    },
                                    title: {
                                        display: true,
                                        text: 'Percentage',
                                        color: textColor
                                    }
                                },
                                x: {
                                    ticks: {
                                        color: textColor
                                    },
                                    grid: {
                                        color: gridColor
                                    },
                                    title: {
                                        display: true,
                                        text: 'Driver ID',
                                        color: textColor
                                    }
                                }
                            },
                            plugins: {
                                legend: {
                                    labels: {
                                        color: textColor
                                    }
                                },
                                title: {
                                    display: true,
                                    text: 'Driver Alert Percentages',
                                    color: textColor
                                }
                            }
                        }
                    });
                }
            }
            
            function updateViolationsTable() {
                const tableBody = document.getElementById('violationsTableBody');
                tableBody.innerHTML = '';
                
                const sortedDrivers = Object.entries(driverViolations)
                    .sort((a, b) => (b[1].drowsiness + b[1].alcohol) - (a[1].drowsiness + a[1].alcohol));
                
                for (const [driverId, data] of sortedDrivers) {
                    const row = document.createElement('tr');
                    
                    const drowsyPercent = driverFrameCounts[driverId] ? 
                        ((driverFrameCounts[driverId].drowsy / driverFrameCounts[driverId].total) * 100).toFixed(1) + '%' : '0%';
                        
                    const alcoholPercent = driverFrameCounts[driverId] ? 
                        ((driverFrameCounts[driverId].alcohol / driverFrameCounts[driverId].total) * 100).toFixed(1) + '%' : '0%';
                    
                    const idCell = document.createElement('td');
                    idCell.textContent = driverId;
                    row.appendChild(idCell);
                    
                    const drowsinessCell = document.createElement('td');
                    drowsinessCell.textContent = data.drowsiness;
                    drowsinessCell.className = data.drowsiness > 0 ? 'violations-count' : '';
                    row.appendChild(drowsinessCell);
                    
                    const alcoholCell = document.createElement('td');
                    alcoholCell.textContent = data.alcohol;
                    alcoholCell.className = data.alcohol > 0 ? 'alcohol-count' : '';
                    row.appendChild(alcoholCell);
                    
                    const statusCell = document.createElement('td');
                    statusCell.textContent = data.status;
                    statusCell.style.color = data.status === 'Active' ? 'var(--normal-color)' : '#888';
                    row.appendChild(statusCell);
                    
                    const lastActiveCell = document.createElement('td');
                    lastActiveCell.textContent = new Date(data.lastActive).toLocaleTimeString();
                    row.appendChild(lastActiveCell);
                    
                    const drowsyPercentCell = document.createElement('td');
                    drowsyPercentCell.textContent = drowsyPercent;
                    drowsyPercentCell.style.color = 'var(--drowsy-color)';
                    drowsyPercentCell.style.fontWeight = 'bold';
                    row.appendChild(drowsyPercentCell);
                    
                    const alcoholPercentCell = document.createElement('td');
                    alcoholPercentCell.textContent = alcoholPercent;
                    alcoholPercentCell.style.color = 'var(--alcohol-color)';
                    alcoholPercentCell.style.fontWeight = 'bold';
                    row.appendChild(alcoholPercentCell);
                    
                    tableBody.appendChild(row);
                }
            }

            document.addEventListener('DOMContentLoaded', function() {
                // Start updating performance data
                updatePerformanceData();
                
                // Apply theme from localStorage
                if (localStorage.getItem('theme') === 'dark') {
                    htmlElement.setAttribute('data-theme', 'dark');
                    themeToggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
                }
            });
        </script>
    </div>
</body>
</html>
"""

@app.route("/")
def stream_viewer():
    return render_template_string(STREAM_VIEWER_HTML)

@app.route("/get_frame")
def get_frame():
    global latest_frames
    requested_driver = request.args.get('driver_id', None)
    
    # If "latest" is requested or no specific driver requested
    if requested_driver == 'latest' or not requested_driver:
        # Find most recently active driver
        if not latest_frames:
            return jsonify({"error": "No frames available"}), 404
            
        # Get most recently updated frame
        latest_driver = max(latest_frames.items(), 
                           key=lambda x: x[1]["timestamp"])[0] if latest_frames else None
                           
        if latest_driver:
            driver_data = latest_frames[latest_driver]
            return jsonify({
                "frame": driver_data["frame"],
                "driver_id": latest_driver,
                "drowsiness_detected": driver_data["drowsiness_detected"],
                "alcohol_detected": driver_data.get("alcohol_detected", False),
                "active_drivers": len(active_drivers)
            })
    
    # If a specific driver is requested
    elif requested_driver in latest_frames:
        driver_data = latest_frames[requested_driver]
        return jsonify({
            "frame": driver_data["frame"],
            "driver_id": requested_driver,
            "drowsiness_detected": driver_data["drowsiness_detected"],
            "alcohol_detected": driver_data.get("alcohol_detected", False),
            "active_drivers": len(active_drivers)
        })
    
    return jsonify({"error": "No matching frames found"}), 404

@app.route("/get_all_frames")
def get_all_frames():
    """Get frames for all active drivers for multi-panel display"""
    global latest_frames, active_drivers
    
    # Filter to only include active drivers
    current_time = time.time()
    active_driver_frames = []
    
    for driver_id, last_active_time in list(active_drivers.items()):
        if current_time - last_active_time <= DRIVER_TIMEOUT:
            if driver_id in latest_frames:
                driver_data = latest_frames[driver_id]
                active_driver_frames.append({
                    "frame": driver_data["frame"],
                    "driver_id": driver_id,
                    "drowsiness_detected": driver_data["drowsiness_detected"],
                    "alcohol_detected": driver_data.get("alcohol_detected", False)
                })
    
    return jsonify({
        "drivers": active_driver_frames,
        "driver_ids": list(active_drivers.keys()),
        "active_count": len(active_drivers),
        "timestamp": current_time
    })

@app.route("/stream", methods=["POST"])
def stream():
    data = request.get_json()
    if not data or "frame" not in data or "driver_id" not in data:
        return jsonify({"error": "Missing required data"}), 400

    response, status_code = processor.process_frame(
        data["driver_id"],
        data["frame"]
    )
    return jsonify(response), status_code

@app.route("/stats")
def get_stats():
    drowsy_drivers = [driver_id for driver_id, data in latest_frames.items() 
                     if data.get("drowsiness_detected", False)]
    
    # Add alcohol detection stats
    alcohol_drivers = [driver_id for driver_id, data in latest_frames.items() 
                     if data.get("alcohol_detected", False)]
    
    return jsonify({
        "active_drivers": len(active_drivers),
        "driver_ids": list(active_drivers.keys()),
        "drowsy_drivers": drowsy_drivers,
        "alcohol_drivers": alcohol_drivers,
        "system_uptime": time.time() - startup_time
    })

@app.route("/analytics")
def analytics_dashboard():
    conn = get_db_connection()
    
    total_sessions = conn.execute("SELECT COUNT(*) FROM driver_sessions").fetchone()[0]
    total_drowsiness_events = conn.execute("SELECT COUNT(*) FROM drowsiness_events").fetchone()[0]
    total_alcohol_events = conn.execute("SELECT COUNT(*) FROM alcohol_events").fetchone()[0]
    
    recent_sessions = conn.execute("""
        SELECT * FROM driver_sessions ORDER BY start_time DESC LIMIT 10
    """).fetchall()
    
    drowsiness_chart = generate_drowsiness_chart()
    alcohol_chart = generate_alcohol_chart()
    
    conn.close()
    
    return render_template_string(ANALYTICS_HTML, 
                                  total_sessions=total_sessions,
                                  total_drowsiness_events=total_drowsiness_events,
                                  total_alcohol_events=total_alcohol_events,
                                  recent_sessions=recent_sessions,
                                  drowsiness_chart=drowsiness_chart,
                                  alcohol_chart=alcohol_chart)

@app.route("/system_status")
def system_status():
    """Check if required files are available"""
    status = {
        "alert_sound": os.path.exists(ALERT_SOUND_PATH),
        "alcohol_alert_sound": os.path.exists(ALCOHOL_ALERT_SOUND_PATH),
        "shape_predictor": os.path.exists("shape_predictor_68_face_landmarks.dat"),
        "active_drivers": len(active_drivers),
    }
    return jsonify(status)

@app.route("/driver_performance_data")
def driver_performance_data():
    """Get driver performance data for analytics dashboard"""
    global latest_frames, active_drivers
    
    # Convert driver violations to serializable format
    violations_data = {}
    frame_counts_data = {}
    
    # Include both active and recently inactive drivers (last 1 hour)
    active_cutoff = time.time() - 3600
    
    # Get all drivers that have been active in the last hour
    conn = get_db_connection()
    recent_drivers = conn.execute("""
        SELECT DISTINCT driver_id FROM driver_sessions 
        WHERE start_time > datetime('now', '-1 hour')
    """).fetchall()
    conn.close()
    
    all_drivers = set(active_drivers.keys())
    for row in recent_drivers:
        all_drivers.add(row['driver_id'])
    
    for driver_id in all_drivers:
        # Get basic stats from database
        conn = get_db_connection()
        
        # Get drowsiness events count
        drowsiness_count = conn.execute(
            "SELECT COUNT(*) FROM drowsiness_events WHERE driver_id = ? AND event_type = 'drowsiness_detected'", 
            (driver_id,)
        ).fetchone()[0]
        
        # Get alcohol events count
        alcohol_count = conn.execute(
            "SELECT COUNT(*) FROM alcohol_events WHERE driver_id = ? AND event_type = 'alcohol_detected'", 
            (driver_id,)
        ).fetchone()[0]
        
        # Get session data
        session_data = conn.execute(
            "SELECT SUM(frames_processed) as total_frames, SUM(drowsiness_detected_count) as drowsy_frames, " +
            "SUM(alcohol_detected_count) as alcohol_frames FROM driver_sessions WHERE driver_id = ?",
            (driver_id,)
        ).fetchone()
        
        conn.close()
        
        # Determine if driver is currently active
        is_active = driver_id in active_drivers and (time.time() - active_drivers[driver_id]) <= DRIVER_TIMEOUT
        status = 'Active' if is_active else 'Inactive'
        last_active = active_drivers.get(driver_id, active_cutoff)
        
        # Add to violations data
        violations_data[driver_id] = {
            'drowsiness': drowsiness_count,
            'alcohol': alcohol_count,
            'status': status,
            'lastActive': last_active * 1000  # Convert to milliseconds for JS
        }
        
        # Add frame counts data
        total_frames = session_data['total_frames'] if session_data and session_data['total_frames'] else 100
        drowsy_frames = session_data['drowsy_frames'] if session_data and session_data['drowsy_frames'] else drowsiness_count
        alcohol_frames = session_data['alcohol_frames'] if session_data and session_data['alcohol_frames'] else alcohol_count
        
        frame_counts_data[driver_id] = {
            'total': max(total_frames, 1),  # Avoid division by zero
            'drowsy': drowsy_frames,
            'alcohol': alcohol_frames
        }
    
    return jsonify({
        'violations': violations_data,
        'frameCounts': frame_counts_data,
        'timestamp': time.time()
    })

if __name__ == "__main__":
    startup_time = time.time()
    init_db()
    if SAVE_FRAMES:
        os.makedirs("frames", exist_ok=True)
    cleanup_old_data()
    
    # Generate alert sounds if missing
    generate_default_alert_sounds()
    
    print("[INFO] Starting multi-source drowsiness detection server on 0.0.0.0:5000")
    print("[INFO] Make sure the shape_predictor_68_face_landmarks.dat file is in the current directory")
    print(f"[INFO] Alert sound file: {'Found' if os.path.exists(ALERT_SOUND_PATH) else 'Missing (will generate)'}")
    print(f"[INFO] Alcohol alert sound file: {'Found' if os.path.exists(ALCOHOL_ALERT_SOUND_PATH) else 'Missing (will generate)'}")
    print("[INFO] Access the web interface at http://localhost:5000")
    
    # Install waitress if not already installed
    try:
        from waitress import serve
        print("[INFO] Starting multi-source drowsiness detection server on 0.0.0.0:5000")
        print("[INFO] Make sure the shape_predictor_68_face_landmarks.dat file is in the current directory")
        print("[INFO] Access the web interface at http://localhost:5000")
        print("[INFO] Analytics dashboard available at http://localhost:5000/analytics")
        serve(app, host="0.0.0.0", port=5000, threads=8)
    except ImportError:
        print("[ERROR] Waitress module not installed. Installing...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "waitress"])
        print("[INFO] Waitress installed. Please restart the server.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to start server: {str(e)}")
        # Fall back to Flask's built-in server
        print("[INFO] Falling back to Flask's development server")
        app.run(host="0.0.0.0", port=5000, debug=False)
import os
PORT = int(os.environ.get("PORT", 5000))

from flask import Flask, request, jsonify, render_template_string
import base64
import cv2
import numpy as np
import time
import json
import logging
from threading import Lock
from queue import Queue
import concurrent.futures
import dlib
from scipy.spatial import distance as dist
import os
import pygame
import threading
from typing import Dict, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from file
def load_config():
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            logging.info("Configuration loaded successfully")
            return config
    except FileNotFoundError:
        logging.warning("Config file not found, using defaults")
        return {
            "MAX_WORKERS": 4,
            "FRAME_QUEUE_SIZE": 30,
            "MAX_FRAME_RATE": 30,
            "FRAME_SKIP_THRESHOLD": 0.5,
            "PROCESS_EVERY_N_FRAMES": 1,
            "SAVE_FRAMES": False,
            "EAR_THRESHOLD": 0.22,
            "CONSEC_FRAMES": 10,
            "ALERT_SOUND_PATH": "alert_sound.wav",
            "SOUND_ENABLED": True,
            "DEBUG_MODE": True,
            "DRIVER_TIMEOUT": 30
        }

# Load configuration
CONFIG = load_config()

# Configuration variables from config file
MAX_WORKERS = CONFIG.get("MAX_WORKERS", 4)
FRAME_QUEUE_SIZE = CONFIG.get("FRAME_QUEUE_SIZE", 30)
MAX_FRAME_RATE = CONFIG.get("MAX_FRAME_RATE", 30)
FRAME_SKIP_THRESHOLD = CONFIG.get("FRAME_SKIP_THRESHOLD", 0.5)
PROCESS_EVERY_N_FRAMES = CONFIG.get("PROCESS_EVERY_N_FRAMES", 1)
SAVE_FRAMES = CONFIG.get("SAVE_FRAMES", False)
EAR_THRESHOLD = CONFIG.get("EAR_THRESHOLD", 0.22)
CONSEC_FRAMES = CONFIG.get("CONSEC_FRAMES", 10)
ALERT_SOUND_PATH = CONFIG.get("ALERT_SOUND_PATH", "alert_sound.wav")
SOUND_ENABLED = CONFIG.get("SOUND_ENABLED", True)
DEBUG_MODE = CONFIG.get("DEBUG_MODE", True)
DRIVER_TIMEOUT = CONFIG.get("DRIVER_TIMEOUT", 30)

# Global state for drowsiness detection
drowsy_counter = {}  # Track consecutive frames for each driver
sound_playing = {}   # Track if sound is playing for each driver
active_drivers = {}  # Track active drivers and their last activity time
ear_values = {}      # Track EAR values for debugging

# Initialize pygame for audio playback if sound is enabled
pygame_initialized = False
if SOUND_ENABLED:
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        pygame.init()
        pygame_initialized = True
        logging.info("Pygame initialized successfully")
        
        # Check if sound file exists
        if os.path.exists(ALERT_SOUND_PATH):
            logging.info(f"Alert sound file found at: {ALERT_SOUND_PATH}")
        else:
            logging.error(f"Alert sound file not found at: {ALERT_SOUND_PATH}")
            logging.error("Sound alerts will not work!")
    except Exception as e:
        logging.error(f"Failed to initialize pygame: {str(e)}")

def play_alert_sound(driver_id):
    """Play the alert sound in a loop for a specific driver."""
    if not SOUND_ENABLED or not pygame_initialized:
        logging.warning("Sound is disabled or pygame not initialized")
        return
        
    if driver_id not in sound_playing or not sound_playing[driver_id]:
        try:
            # Use a channel based on driver_id
            channel_id = abs(hash(driver_id)) % 8  # Use up to 8 channels
            
            # Check if sound file exists
            if not os.path.exists(ALERT_SOUND_PATH):
                logging.error(f"Alert sound file not found at: {ALERT_SOUND_PATH}")
                return
                
            # Load and play sound
            sound = pygame.mixer.Sound(ALERT_SOUND_PATH)
            channel = pygame.mixer.Channel(channel_id)
            channel.play(sound, loops=-1)  # Loop indefinitely
            
            sound_playing[driver_id] = True
            logging.info(f"Alert sound started for driver: {driver_id}")
        except Exception as e:
            logging.error(f"Error playing alert sound: {str(e)}")

def stop_alert_sound(driver_id):
    """Stop the alert sound for a specific driver."""
    if not SOUND_ENABLED or not pygame_initialized:
        return
        
    if driver_id in sound_playing and sound_playing[driver_id]:
        try:
            channel_id = abs(hash(driver_id)) % 8
            channel = pygame.mixer.Channel(channel_id)
            channel.stop()
            sound_playing[driver_id] = False
            logging.info(f"Alert sound stopped for driver: {driver_id}")
        except Exception as e:
            logging.error(f"Error stopping alert sound: {str(e)}")

def calculate_EAR(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) to detect eye closure.
    :param eye: Array of coordinates for the eye landmarks.
    :return: EAR value.
    """
    try:
        A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
        B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
        C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
        
        # Avoid division by zero
        if C < 0.1:  # If the eyes are too close together
            return 0
            
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        logging.error(f"Error calculating EAR: {str(e)}")
        return 0

def cleanup_inactive_drivers():
    """Remove inactive drivers from tracking dictionaries."""
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
        if driver_id in ear_values:
            del ear_values[driver_id]
        
        logging.info(f"Driver {driver_id} marked as inactive and removed from tracking")

class FrameProcessor:
    def __init__(self):
        self.frame_queues = {}
        self.last_frame_times = {}
        self.frame_counters = {}
        self.lock = Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Load dlib's face detector and shape predictor
        logging.info("Loading facial landmark predictor...")
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Shape predictor file not found: {model_path}")
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        
        # Eye landmark indices
        self.lStart, self.lEnd = 42, 48  # Left eye
        self.rStart, self.rEnd = 36, 42  # Right eye
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._run_cleanup, daemon=True)
        self.cleanup_thread.start()

    def _run_cleanup(self):
        """Periodically clean up inactive drivers."""
        while True:
            cleanup_inactive_drivers()
            time.sleep(10)  # Check every 10 seconds

    def process_frame(self, driver_id: str, frame_data: str) -> Tuple[Dict, int]:
        """Process incoming frame with smart queue management"""
        current_time = time.time()
        
        with self.lock:
            # Initialize for new drivers
            if driver_id not in self.frame_queues:
                self.frame_queues[driver_id] = Queue(maxsize=FRAME_QUEUE_SIZE)
                self.frame_counters[driver_id] = 0
                drowsy_counter[driver_id] = 0
                sound_playing[driver_id] = False
                ear_values[driver_id] = []
                logging.info(f"New driver registered: {driver_id}")
            
            # Update active drivers tracking
            active_drivers[driver_id] = current_time
            
            # Implement frame skipping for performance
            self.frame_counters[driver_id] = (self.frame_counters[driver_id] + 1) % PROCESS_EVERY_N_FRAMES
            
            # Skip processing if queue is backing up or not the right frame in sequence
            queue = self.frame_queues[driver_id]
            if queue.qsize() > FRAME_QUEUE_SIZE * FRAME_SKIP_THRESHOLD and self.frame_counters[driver_id] != 0:
                return {"status": "skipped_for_performance"}, 202

            # Update last processed time
            self.last_frame_times[driver_id] = current_time

        # Submit for processing
        future = self.executor.submit(self._decode_and_process, driver_id, frame_data)
        return {"status": "processing"}, 202

    def _decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """Decode base64 frame with error handling and optimization"""
        try:
            frame_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            # Use regular decoding for better quality
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            # Resize for consistent processing and performance
            return cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logging.error(f"Frame decoding error: {str(e)}")
            return None

    def _decode_and_process(self, driver_id: str, frame_data: str) -> None:
        """Decode and process frame in a thread-safe manner"""
        try:
            # Decode frame
            img = self._decode_frame(frame_data)
            if img is None:
                return

            # Process frame with drowsiness detection
            processed, drowsiness_detected, ear = self._custom_processing(driver_id, img)
            
            # Store latest frame data with compression
            global latest_frame, latest_driver_id, latest_drowsiness_status
            _, buffer = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            latest_frame = base64.b64encode(buffer).decode('utf-8')
            latest_driver_id = driver_id
            latest_drowsiness_status = drowsiness_detected
            
            # Keep track of recent EAR values for debugging
            if len(ear_values[driver_id]) > 10:
                ear_values[driver_id].pop(0)
            ear_values[driver_id].append(ear)
            
            # Debug log
            if DEBUG_MODE and drowsiness_detected:
                logging.info(f"Driver {driver_id} - DROWSINESS DETECTED! EAR: {ear:.3f}")
            
            # Save frame if enabled and drowsiness detected
            if SAVE_FRAMES and drowsiness_detected:
                self._save_frame(driver_id, processed)

        except Exception as e:
            logging.error(f"Error processing frame from {driver_id}: {str(e)}")

    def _custom_processing(self, driver_id: str, frame: np.ndarray) -> Tuple[np.ndarray, bool, float]:
        """
        Process the frame with drowsiness detection
        :return: Tuple of (processed frame, drowsiness detected flag, ear value)
        """
        # Make a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for low-light conditions
        gray = cv2.equalizeHist(gray)
        
        # Detect faces in the frame
        faces = self.detector(gray, 0)
        
        # Initialize drowsiness status
        drowsiness_detected = False
        current_ear = 0.0
        
        # Draw face count
        cv2.putText(display_frame, f"Faces: {len(faces)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw current threshold value
        cv2.putText(display_frame, f"EAR Threshold: {EAR_THRESHOLD}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
        # Draw driver ID
        cv2.putText(display_frame, f"Driver ID: {driver_id[:10]}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # If no faces are detected, reset counter
        if len(faces) == 0:
            # Only reset after several frames without faces
            drowsy_counter[driver_id] = max(0, drowsy_counter[driver_id] - 1)
        
        for face in faces:
            # Predict facial landmarks
            shape = self.predictor(gray, face)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Extract eye landmarks
            leftEye = shape_np[self.lStart:self.lEnd]
            rightEye = shape_np[self.rStart:self.rEnd]
            
            # Compute EAR for both eyes
            leftEAR = calculate_EAR(leftEye)
            rightEAR = calculate_EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            current_ear = ear
            
            # Draw bounding box around face
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw contours around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(display_frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(display_frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Check if EAR is below the threshold
            if ear < EAR_THRESHOLD:
                drowsy_counter[driver_id] += 1
                if drowsy_counter[driver_id] >= CONSEC_FRAMES:
                    # Display alert on screen
                    cv2.putText(display_frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Set the drowsiness flag
                    drowsiness_detected = True
                    
                    # Play alert sound if not already playing
                    if not sound_playing.get(driver_id, False):
                        play_alert_sound(driver_id)
                        
                    # Debug counter
                    if DEBUG_MODE:
                        cv2.putText(display_frame, f"Counter: {drowsy_counter[driver_id]}", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            else:
                # Reset counter gradually to avoid flicker
                drowsy_counter[driver_id] = max(0, drowsy_counter[driver_id] - 1)
                
                # Stop alert sound only when counter is fully reset
                if drowsy_counter[driver_id] == 0 and sound_playing.get(driver_id, False):
                    stop_alert_sound(driver_id)
                    
                # Debug counter
                if DEBUG_MODE:
                    cv2.putText(display_frame, f"Counter: {drowsy_counter[driver_id]}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display EAR value
            cv2.putText(display_frame, f"EAR: {ear:.3f}", (480, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame, drowsiness_detected, current_ear

    def _save_frame(self, driver_id: str, frame: np.ndarray) -> None:
        """Save frame to disk for debugging"""
        timestamp = int(time.time() * 1000)
        os.makedirs(f"frames/{driver_id}", exist_ok=True)
        cv2.imwrite(f"frames/{driver_id}/{timestamp}.jpg", frame)

# Initialize Flask app and processor
app = Flask(__name__)
processor = FrameProcessor()

# Global variables to store the latest frame and data
latest_frame = None
latest_driver_id = None
latest_drowsiness_status = False

# HTML template for the stream viewer with multi-driver support
STREAM_VIEWER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Drowsiness Detection System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0; 
            padding: 0; 
            background-color: #f0f2f5;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        header {
            background-color: #1e3a8a;
            color: white;
            padding: 15px 0;
            text-align: center;
            border-radius: 8px 8px 0 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 { 
            margin: 0; 
            font-size: 28px;
        }
        .main-content {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .video-panel {
            flex: 2;
            min-width: 640px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .controls-panel {
            flex: 1;
            min-width: 300px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .stream-container { 
            position: relative;
        }
        img { 
            width: 100%; 
            display: block;
        }
        .status-bar { 
            padding: 15px; 
            font-weight: bold; 
            text-align: center;
            border-top: 1px solid #eee;
            transition: all 0.5s ease;
        }
        .alert { 
            background-color: #ff4d4d; 
            color: white; 
            animation: pulse 1.5s infinite;
        }
        .normal { 
            background-color: #4CAF50; 
            color: white; 
        }
        @keyframes pulse {
            0% { background-color: #ff4d4d; }
            50% { background-color: #ff0000; }
            100% { background-color: #ff4d4d; }
        }
        .driver-selector {
            margin-bottom: 20px;
        }
        select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            background-color: white;
        }
        .stats {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }
        .stat-item {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stat-label {
            font-weight: 500;
            color: #555;
        }
        .stat-value {
            font-weight: 600;
            color: #1e3a8a;
        }
        .info-section {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9f5fe;
            border-radius: 8px;
            border-left: 5px solid #1e88e5;
        }
        .info-section h2 {
            margin-top: 0;
            color: #1e88e5;
        }
        footer {
            margin-top: 30px;
            text-align: center;
            font-size: 14px;
            color: #666;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Drowsiness Detection System</h1>
        </header>
        
        <div class="main-content">
            <div class="video-panel">
                <div class="stream-container">
                    <img id="streamImage" src="" alt="Waiting for video...">
                    <div style="text-align: center; padding: 20px; color: #666;" id="noVideoMessage">Waiting for video feed...</div>
                </div>
                <div id="statusBar" class="status-bar normal">Driver Status: Waiting for data</div>
            </div>
            
            <div class="controls-panel">
                <div class="driver-selector">
                    <label for="driverSelect" style="display: block; margin-bottom: 5px; font-weight: 500;">Select Driver:</label>
                    <select id="driverSelect">
                        <option value="latest">Latest Active Driver</option>
                        <!-- Additional drivers will be populated dynamically -->
                    </select>
                </div>
                
                <div class="stats">
                    <div class="stat-item">
                        <span class="stat-label">Driver ID:</span>
                        <span class="stat-value" id="driverId">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Frame Rate:</span>
                        <span class="stat-value" id="frameRate">0 fps</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Latency:</span>
                        <span class="stat-value" id="latency">0 ms</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Active Drivers:</span>
                        <span class="stat-value" id="activeDrivers">0</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Eye Aspect Ratio:</span>
                        <span class="stat-value" id="earValue">0.00</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Alert Status:</span>
                        <span class="stat-value" id="alertStatus">Normal</span>
                    </div>
                </div>
                
                <div class="info-section">
                    <h2>System Information</h2>
                    <p>This system monitors driver alertness in real-time using computer vision analysis. When signs of drowsiness are detected, an alert is triggered.</p>
                    <p>EAR Threshold: <strong id="earThreshold">${EAR_THRESHOLD}</strong></p>
                    <p>Consecutive Frames: <strong>${CONSEC_FRAMES}</strong></p>
                </div>
            </div>
        </div>
        
        <footer>
            &copy; 2025 Drowsiness Detection System
        </footer>
    </div>
    
    <script>
        // Variables to track frame rate and latency
        let frameCount = 0;
        let lastFrameTime = Date.now();
        let frameRates = [];
        let knownDrivers = new Set();
        let earValues = [];
        let noVideoTimeout;
        
        // Show initial loading message
        document.getElementById('noVideoMessage').style.display = 'block';
        
        function updateStream() {
            const selectedDriver = document.getElementById('driverSelect').value;
            const startTime = Date.now();
            
            fetch('/get_frame' + (selectedDriver !== 'latest' ? '?driver_id=' + selectedDriver : ''))
                .then(response => {
                    if (!response.ok) {
                        throw new Error('No frame available');
                    }
                    return response.json();
                })
                .then(data => {
                    // Clear the no-video timeout
                    clearTimeout(noVideoTimeout);
                    
                    // Calculate latency
                    const latency = Date.now() - startTime;
                    document.getElementById('latency').textContent = `${latency} ms`;
                    
                    if (data.frame) {
                        // Hide the no video message
                        document.getElementById('noVideoMessage').style.display = 'none';
                        
                        // Update video stream
                        document.getElementById('streamImage').src = 'data:image/jpeg;base64,' + data.frame;
                        
                        // Update driver ID
                        if (data.driver_id) {
                            document.getElementById('driverId').textContent = data.driver_id;
                            
                            // Add driver to selector if new
                            if (!knownDrivers.has(data.driver_id)) {
                                knownDrivers.add(data.driver_id);
                                const option = document.createElement('option');
                                option.value = data.driver_id;
                                option.textContent = 'Driver: ' + data.driver_id;
                                document.getElementById('driverSelect').appendChild(option);
                            }
                        }
                        
                        // Update drowsiness status
                        const statusBar = document.getElementById('statusBar');
                        const alertStatus = document.getElementById('alertStatus');
                        
                        if (data.drowsiness_detected) {
                            statusBar.className = 'status-bar alert';
                            statusBar.textContent = '⚠️ DROWSINESS ALERT! ⚠️';
                            alertStatus.textContent = 'ALERT';
                            alertStatus.style.color = '#ff0000';
                        } else {
                            statusBar.className = 'status-bar normal';
                            statusBar.textContent = 'Driver Status: Alert';
                            alertStatus.textContent = 'Normal';
                            alertStatus.style.color = '#4CAF50';
                        }
                        
                        // Update EAR value if available
                        if (data.ear) {
                            document.getElementById('earValue').textContent = data.ear.toFixed(3);
                            earValues.push(data.ear);
                            if (earValues.length > 10) earValues.shift();
                        }
                        
                        // Update active drivers count
                        if (data.active_drivers !== undefined) {
                            document.getElementById('activeDrivers').textContent = data.active_drivers;
                        }
                        
                        // Calculate frame rate
                        frameCount++;
                        const now = Date.now();
                        if (now - lastFrameTime >= 1000) { // Update every second
                            const fps = frameCount / ((now - lastFrameTime) / 1000);
                            frameRates.push(fps);
                            if (frameRates.length > 5) frameRates.shift(); // Keep last 5 readings
                            
                            // Calculate average FPS
                            const avgFps = frameRates.reduce((a, b) => a + b, 0) / frameRates.length;
                            document.getElementById('frameRate').textContent = `${avgFps.toFixed(1)} fps`;
                            
                            frameCount = 0;
                            lastFrameTime = now;
                        }
                    }
                    
                    // Schedule next update
                    setTimeout(updateStream, 33); // ~30 fps update rate
                })
                .catch(error => {
                    console.error('Error:', error);
                    
                    // Show no video message after a short delay
                    noVideoTimeout = setTimeout(() => {
                        document.getElementById('noVideoMessage').style.display = 'block';
                    }, 3000);
                    
                    setTimeout(updateStream, 1000); // Retry after 1s on error
                });
        }
        
        // Start the stream when page loads
        document.addEventListener('DOMContentLoaded', function() {
            updateStream();
        });
    </script>
</body>
</html>
"""

@app.route("/")
def stream_viewer():
    """Endpoint to view the live stream"""
    return render_template_string(STREAM_VIEWER_HTML)

@app.route("/get_frame")
def get_frame():
    """Endpoint to get the latest frame, optionally filtered by driver_id"""
    global latest_frame, latest_driver_id, latest_drowsiness_status
    
    # Check for specific driver request
    requested_driver = request.args.get('driver_id', None)
    
    # If we have frames and either no specific driver is requested or it matches
    if latest_frame and (not requested_driver or requested_driver == latest_driver_id):
        # Get current EAR value for this driver
        driver_ear = 0.0
        if latest_driver_id in ear_values and ear_values[latest_driver_id]:
            driver_ear = ear_values[latest_driver_id][-1]
            
        return jsonify({
            "frame": latest_frame,
            "driver_id": latest_driver_id,
            "drowsiness_detected": latest_drowsiness_status,
            "active_drivers": len(active_drivers),
            "ear": driver_ear
        })
    
    # No matching frames
    return jsonify({"error": "No matching frames found"}), 404

@app.route("/stream", methods=["POST"])
def stream():
    """Endpoint for receiving video frames"""
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
    """Endpoint to get system statistics"""
    stats = {
        "active_drivers": len(active_drivers),
        "driver_ids": list(active_drivers.keys()),
        "system_uptime": time.time() - startup_time,
        "ear_threshold": EAR_THRESHOLD,
        "consecutive_frames": CONSEC_FRAMES,
        "frame_rate": MAX_FRAME_RATE,
        "sound_enabled": SOUND_ENABLED and pygame_initialized,
        "sound_file_exists": os.path.exists(ALERT_SOUND_PATH)
    }
    
    # Add driver-specific stats
    driver_stats = {}
    for driver_id in active_drivers:
        driver_stats[driver_id] = {
            "drowsy_counter": drowsy_counter.get(driver_id, 0),
            "sound_playing": sound_playing.get(driver_id, False),
            "ear_values": ear_values.get(driver_id, [])
        }
    
    stats["drivers"] = driver_stats
    return jsonify(stats)

@app.route("/settings", methods=["GET", "POST"])
def settings():
    """Get or update settings"""
    global EAR_THRESHOLD, CONSEC_FRAMES, SOUND_ENABLED
    
    if request.method == "POST":
        data = request.get_json()
        if data:
            if "ear_threshold" in data:
                EAR_THRESHOLD = float(data["ear_threshold"])
            if "consec_frames" in data:
                CONSEC_FRAMES = int(data["consec_frames"])
            if "sound_enabled" in data:
                SOUND_ENABLED = bool(data["sound_enabled"])
            return jsonify({"status": "settings updated"})
    
    # GET request returns current settings
    return jsonify({
        "ear_threshold": EAR_THRESHOLD,
        "consec_frames": CONSEC_FRAMES,
        "sound_enabled": SOUND_ENABLED
    })

if __name__ == "__main__":
    # Record startup time
    startup_time = time.time()
    
    # Create frames directory if needed
    if SAVE_FRAMES:
        os.makedirs("frames", exist_ok=True)
    
    # Check for the sound file
    if SOUND_ENABLED and not os.path.exists(ALERT_SOUND_PATH):
        logging.warning(f"Alert sound file not found at: {ALERT_SOUND_PATH}")
        print(f"\n[WARNING] Alert sound file not found: {ALERT_SOUND_PATH}")
        print("Please create or download a WAV file for alerts.")
        print("Example command to create a simple alert sound using ffmpeg:")
        print(f'ffmpeg -f lavfi -i "sine=frequency=1000:duration=2" {ALERT_SOUND_PATH}\n')
    
    # Run production-ready server
    from waitress import serve
    print("[INFO] Starting drowsiness detection server on 0.0.0.0:5000")
    print("[INFO] Make sure the shape_predictor_68_face_landmarks.dat file is in the current directory")
    print("[INFO] Access the web interface at http://localhost:5000")
    serve(app, host="0.0.0.0", port=5000, threads=8)

    if __name__ == "__main__":
    # Record startup time
      startup_time = time.time()
    
    # Create frames directory if needed
    if SAVE_FRAMES:
        os.makedirs("frames", exist_ok=True)
    
    # Check for the sound file
    if SOUND_ENABLED and not os.path.exists(ALERT_SOUND_PATH):
        logging.warning(f"Alert sound file not found at: {ALERT_SOUND_PATH}")
        print(f"\n[WARNING] Alert sound file not found: {ALERT_SOUND_PATH}")
        print("Creating a simple alert sound...")
        try:
            # Generate a simple tone using numpy
            import numpy as np
            from scipy.io.wavfile import write
            
            sample_rate = 44100
            duration = 2  # seconds
            frequency = 1000  # Hz
            t = np.linspace(0, duration, int(sample_rate * duration))
            wave = 0.5 * np.sin(2 * np.pi * frequency * t)
            write(ALERT_SOUND_PATH, sample_rate, wave.astype(np.float32))
            print(f"Created alert sound at {ALERT_SOUND_PATH}")
        except Exception as e:
            print(f"Failed to create alert sound: {e}")
    
    # Run production-ready server
    from waitress import serve
    print(f"[INFO] Starting drowsiness detection server on port {PORT}")
    print("[INFO] Make sure the shape_predictor_68_face_landmarks.dat file is in the current directory")
    serve(app, host="0.0.0.0", port=PORT, threads=8)
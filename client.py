import cv2
import base64
import requests
import time
import argparse
import uuid
import json
import os
import sys
from threading import Thread, Event
import threading
from queue import Queue, Empty  # Fixed import to use Empty from queue
from datetime import datetime

# Performance monitoring
class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.frame_times = []
        self.window_size = window_size
        self.frames_sent = 0
        self.frames_dropped = 0
        self.start_time = time.time()
        
    def update_sent(self):
        self.frames_sent += 1
        self.frame_times.append(time.time())
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            
    def update_dropped(self):
        self.frames_dropped += 1
        
    def get_fps(self):
        if len(self.frame_times) < 2:
            return 0
        # Calculate FPS based on recent frames
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff <= 0:
            return 0
        return (len(self.frame_times) - 1) / time_diff
        
    def get_stats(self):
        elapsed = time.time() - self.start_time
        total_fps = self.frames_sent / elapsed if elapsed > 0 else 0
        
        return {
            "frames_sent": self.frames_sent,
            "frames_dropped": self.frames_dropped,
            "current_fps": self.get_fps(),
            "average_fps": total_fps,
            "runtime": elapsed
        }

def frame_producer(cap, frame_queue, stop_event, monitor):
    """Captures frames from the camera and puts them in the queue"""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to capture frame from camera")
            time.sleep(0.1)
            continue
            
        # If queue is full, drop the frame
        if frame_queue.full():
            monitor.update_dropped()
        else:
            current_time = time.time()
            frame_queue.put((frame, current_time))
            
        # Control capture rate based on the camera's capabilities
        time.sleep(0.01)  # ~100 FPS max, will be limited by camera hardware

def frame_consumer(frame_queue, session, args, driver_id, stop_event, monitor, display_event):
    """Processes frames from the queue and sends them to the server"""
    frame_counter = 0
    retry_count = 0
    last_display_frame = None
    
    while not stop_event.is_set():
        try:
            # Get a frame from the queue or wait up to 100ms
            try:
                frame_data = frame_queue.get(timeout=0.1)
                frame = frame_data[0]
                capture_time = frame_data[1]
            except Empty:
                continue
                
            # Skip frames based on processing capabilities
            frame_counter = (frame_counter + 1) % args.process_every
            if frame_counter != 0 and not args.preview:
                continue

            # Save a copy for local display if enabled
            if args.preview:
                last_display_frame = frame.copy()
                
            # Resize and optimize the frame for network transmission
            if args.optimize:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                
            # Apply preprocessing if enabled - COMPLETELY REWRITTEN SECTION
            if args.enhance:
                if len(frame.shape) == 3:  # Color image
                    # Alternative approach: Process each channel individually
                    # Split into BGR channels
                    b, g, r = cv2.split(frame)
                    
                    # Apply histogram equalization to each channel
                    b_eq = cv2.equalizeHist(b)
                    g_eq = cv2.equalizeHist(g)
                    r_eq = cv2.equalizeHist(r)
                    
                    # Merge back
                    frame = cv2.merge([b_eq, g_eq, r_eq])
                else:  # Grayscale
                    frame = cv2.equalizeHist(frame)
            
            # JPEG encode with quality setting
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Rest of code remains unchanged...
            # Add timestamp information
            timestamp = datetime.now().isoformat()
            
            # Send frame to server with retry mechanism
            try:
                response = session.post(
                    args.server_url,
                    json={
                        "driver_id": driver_id,
                        "frame": jpg_as_text,
                        "timestamp": timestamp,
                        "client_info": {
                            "version": "2.0.0",
                            "camera_id": args.camera
                        }
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=args.timeout
                )
                
                # Handle response status
                if response.status_code == 429:  # Rate limited
                    monitor.update_dropped()
                    time.sleep(0.1)
                elif response.status_code == 503:  # Queue full
                    monitor.update_dropped()
                    time.sleep(0.2)
                elif response.status_code == 202:  # Accepted
                    monitor.update_sent()
                    retry_count = 0  # Reset retry counter on success
                else:
                    print(f"[WARNING] Server returned status code {response.status_code}: {response.text}")
                    retry_count += 1
            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Request failed: {str(e)}")
                retry_count += 1
                time.sleep(min(retry_count * 0.5, 5))  # Exponential backoff with cap
                
            # Update the display frame if preview is enabled
            if args.preview and last_display_frame is not None:
                # Calculate latency
                latency = int((time.time() - capture_time) * 1000)
                
                # Get performance stats
                stats = monitor.get_stats()
                
                # Add overlay information
                cv2.putText(last_display_frame, f"Driver: {driver_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(last_display_frame, f"FPS: {stats['current_fps']:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(last_display_frame, f"Latency: {latency}ms", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(last_display_frame, f"Sent: {stats['frames_sent']}, Dropped: {stats['frames_dropped']}", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add timestamp
                time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(last_display_frame, time_str, 
                            (10, last_display_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Send to display thread via event
                if not hasattr(display_event, 'frame'):  # Add safety check
                    display_event.frame = None
                display_event.frame = last_display_frame
                display_event.set()
                
        except Exception as e:
            print(f"[ERROR] Unexpected error in frame consumer: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            time.sleep(0.5)

def display_thread_function(display_event, stop_event):
    """Handles displaying frames in a separate thread"""
    window_name = "Drowsiness Detection Client - Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while not stop_event.is_set():
        if display_event.is_set() and hasattr(display_event, 'frame') and display_event.frame is not None:
            cv2.imshow(window_name, display_event.frame)
            display_event.clear()
            
        # Check for key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
            
        # Short sleep to reduce CPU usage
        time.sleep(0.01)

def main():
    parser = argparse.ArgumentParser(description="Drowsiness Detection Client")
    parser.add_argument("--server-url", type=str, default="http://localhost:5000/stream",
                        help="URL of the drowsiness detection server")
    parser.add_argument("--driver-id", type=str, default=None,
                        help="Unique identifier for the driver (defaults to random UUID)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index to use (default: 0)")
    parser.add_argument("--resolution", type=str, default="640x480",
                        help="Camera resolution in format WIDTHxHEIGHT")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target camera FPS (default: 30)")
    parser.add_argument("--preview", action="store_true",
                        help="Show preview window")
    parser.add_argument("--optimize", action="store_true", default=True,
                        help="Apply optimization to frames before sending")
    parser.add_argument("--enhance", action="store_true", default=True,
                        help="Enhance image quality for better detection")
    parser.add_argument("--quality", type=int, default=85,
                        help="JPEG encoding quality (0-100, default: 85)")
    parser.add_argument("--process-every", type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="Server request timeout in seconds (default: 2.0)")
    args = parser.parse_args()

    # Generate driver ID if not provided
    driver_id = args.driver_id if args.driver_id else str(uuid.uuid4())
    print(f"[INFO] Starting client with driver ID: {driver_id}")

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print(f"[WARNING] Invalid resolution format: {args.resolution}. Using default 640x480.")
        width, height = 640, 480

    # Initialize performance monitoring
    monitor = PerformanceMonitor()

    # Open the camera
    print(f"[INFO] Opening camera {args.camera} at {width}x{height} @ {args.fps} FPS")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera. Please check your camera connection.")
        sys.exit(1)

    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering for lower latency
    
    # Try to set MJPG format for better performance if available
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except:
        print("[WARNING] Could not set MJPG format, using default")

    # Check actual camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[INFO] Camera configured at {actual_width}x{actual_height} @ {actual_fps} FPS")
    print(f"[INFO] Enhance mode: {args.enhance}, Quality: {args.quality}, Process every: {args.process_every}")

    # Create a session for better HTTP performance
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=1,
        pool_maxsize=4,
        max_retries=0,
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # Create thread synchronization objects
    frame_queue = Queue(maxsize=30)
    stop_event = threading.Event()
    display_event = threading.Event()
    display_event.frame = None  # Initialize the frame attribute properly

    # Start threads
    producer = Thread(target=frame_producer, args=(cap, frame_queue, stop_event, monitor), daemon=True)
    consumer = Thread(target=frame_consumer, args=(frame_queue, session, args, driver_id, stop_event, monitor, display_event), daemon=True)
    
    producer.start()
    consumer.start()
    
    # Start display thread if preview is enabled
    if args.preview:
        display_thread = Thread(target=display_thread_function, args=(display_event, stop_event), daemon=True)
        display_thread.start()
        print("[INFO] Preview window enabled. Press 'q' to quit.")
    else:
        print("[INFO] Running in headless mode. Press Ctrl+C to quit.")

    try:
        # Test server connection before starting
        print("[INFO] Testing connection to server...")
        try:
            server_base_url = args.server_url.rsplit('/', 1)[0]  # Remove the last part of the URL
            test_resp = session.get(f"{server_base_url}/stats", timeout=2.0)
            print(f"[INFO] Server connection successful: {test_resp.status_code}")
        except Exception as e:
            print(f"[WARNING] Could not connect to server: {str(e)}")
        
        # Print stats periodically
        while not stop_event.is_set():
            time.sleep(5)
            stats = monitor.get_stats()
            print(f"[STATS] FPS: {stats['current_fps']:.1f}, " + 
                  f"Sent: {stats['frames_sent']}, Dropped: {stats['frames_dropped']}, " +
                  f"Runtime: {int(stats['runtime'])}s")
            
            # If no frames sent after 20 seconds, warn the user
            if stats['runtime'] > 20 and stats['frames_sent'] == 0:
                print("[WARNING] No frames sent after 20 seconds. Check server connection or camera.")
            
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] Unexpected error in main thread: {str(e)}")
    finally:
        # Clean shutdown
        stop_event.set()
        
        # If preview was enabled, wait for display thread
        if args.preview and 'display_thread' in locals():
            display_thread.join(timeout=1.0)
            
        # Wait for other threads
        producer.join(timeout=1.0)
        consumer.join(timeout=1.0)
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        session.close()
        
        # Final statistics
        stats = monitor.get_stats()
        print("\n[SUMMARY]")
        print(f"Total frames sent: {stats['frames_sent']}")
        print(f"Total frames dropped: {stats['frames_dropped']}")
        print(f"Average FPS: {stats['average_fps']:.2f}")
        print(f"Total runtime: {int(stats['runtime'])} seconds")
        print(f"Driver ID: {driver_id}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[CRITICAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
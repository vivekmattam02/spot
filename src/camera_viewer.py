"""
ZED Camera Web Stream with YOLO Detection
==========================================
Streams the ZED camera feed with person detection overlay to a web browser.
Access at http://<apollo-ip>:5000 from any device on the network.
"""

from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from perception import ZEDCamera, PersonDetector

app = Flask(__name__)

# Global objects (initialized on first request)
zed_camera = None
detector = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Spot Camera Feed</title>
    <style>
        body {
            background: #1a1a2e;
            color: #eee;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            margin: 0;
        }
        h1 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .info {
            color: #888;
            margin-bottom: 20px;
        }
        img {
            border: 3px solid #00d4ff;
            border-radius: 8px;
            max-width: 100%;
        }
        .status {
            margin-top: 15px;
            padding: 10px 20px;
            background: #2d2d44;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>🤖 Spot Person Follower - Camera Feed</h1>
    <div class="info">ZED 2i Camera with YOLOv8 Person Detection</div>
    <img src="/video_feed" alt="Camera Feed">
    <div class="status">
        <strong>Stream active</strong> - Refresh page if frozen
    </div>
</body>
</html>
"""


def init_camera_and_detector():
    """Initialize ZED camera and YOLO detector."""
    global zed_camera, detector
    
    if zed_camera is None:
        print("Initializing ZED camera...")
        zed_camera = ZEDCamera(resolution="HD720", fps=30)
        if not zed_camera.open():
            raise RuntimeError("Failed to open ZED camera")
        print("ZED camera opened successfully")
    
    if detector is None:
        print("Initializing YOLO detector...")
        detector = PersonDetector()
        print("YOLO detector ready")


def generate_frames():
    """Generate frames with detection overlay."""
    init_camera_and_detector()
    
    while True:
        # Get frame from ZED camera
        frame = zed_camera.get_frame()
        
        if frame is None:
            # Return a placeholder frame if camera fails
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No frame available", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue
        
        # Make frame contiguous for OpenCV drawing operations
        # ZED camera may return non-contiguous arrays that OpenCV can't draw on
        frame = np.ascontiguousarray(frame)
        
        # Flip frame vertically (ZED camera is mounted upside down)
        frame = cv2.flip(frame, -1)  # -1 = flip both axes (180 degree rotation)
        
        # Run person detection
        detection = detector.detect(frame)
        
        # detector.detect() returns a single Detection object (the best one), not a list
        # Convert to list for iteration
        if detection is not None:
            detections = [detection]
        else:
            detections = []
        
        # Draw detections on frame
        for det in detections:
            # Draw bounding box - bbox is a tuple (x1, y1, x2, y2)
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"Person {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (int(det.center_x), int(det.center_y)), 5, (0, 0, 255), -1)
        
        # Add info overlay
        info_text = f"Detections: {len(detections)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Small delay to control frame rate
        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Serve the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


def cleanup():
    """Cleanup resources."""
    global zed_camera
    if zed_camera is not None:
        zed_camera.close()
        print("ZED camera closed")


if __name__ == '__main__':
    import atexit
    atexit.register(cleanup)
    
    print("=" * 60)
    print("Starting ZED Camera Web Stream")
    print("=" * 60)
    print("Access the stream at: http://<apollo-ip>:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()

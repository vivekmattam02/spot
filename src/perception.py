"""
Perception Module - YOLO Person Detection
==========================================
This module handles real-time person detection using YOLOv8.
It processes camera images and returns bounding box coordinates
for detected persons.

The detector is optimized for the person-following use case:
- Only detects persons (COCO class 0)
- Returns the largest detection when multiple people visible
- Provides confidence scores for filtering
- Runs on GPU when available (RTX 4070 recommended)

Detection Output Format:
-----------------------
Bounding boxes are returned as [x1, y1, x2, y2] where:
    - (x1, y1) = top-left corner
    - (x2, y2) = bottom-right corner
    - Coordinates are in pixels, origin at top-left of image

Author: Generated for Spot Person Follower Project
"""

import logging
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
import numpy as np

# Ultralytics YOLO - provides YOLOv8
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ZED Camera Class
# =============================================================================
class ZEDCamera:
    """
    ZED 2i camera wrapper for capturing frames.
    
    This class provides a simple interface to the ZED camera for the
    person following system.
    
    Usage:
        camera = ZEDCamera()
        camera.open()
        frame = camera.get_frame()
        camera.close()
    """
    
    def __init__(self, resolution: str = "HD720", fps: int = 30):
        """
        Initialize ZED camera.
        
        Args:
            resolution: Camera resolution - "HD720", "HD1080", "HD2K"
            fps: Frames per second (15, 30, 60)
        """
        self.resolution = resolution
        self.fps = fps
        self._camera = None
        self._runtime = None
        self._image = None
        self._is_open = False
        
    def open(self) -> bool:
        """
        Open the ZED camera.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            import pyzed.sl as sl
            
            self._camera = sl.Camera()
            
            # Set initialization parameters
            init_params = sl.InitParameters()
            
            # Set resolution
            if self.resolution == "HD720":
                init_params.camera_resolution = sl.RESOLUTION.HD720
            elif self.resolution == "HD1080":
                init_params.camera_resolution = sl.RESOLUTION.HD1080
            elif self.resolution == "HD2K":
                init_params.camera_resolution = sl.RESOLUTION.HD2K
            else:
                init_params.camera_resolution = sl.RESOLUTION.HD720
                
            init_params.camera_fps = self.fps
            init_params.depth_mode = sl.DEPTH_MODE.NONE  # We don't need depth for detection
            init_params.coordinate_units = sl.UNIT.METER
            
            # Open camera
            status = self._camera.open(init_params)
            
            if status != sl.ERROR_CODE.SUCCESS:
                logger.error(f"Failed to open ZED camera: {status}")
                return False
            
            # Create runtime parameters
            self._runtime = sl.RuntimeParameters()
            
            # Create image container
            self._image = sl.Mat()
            
            self._is_open = True
            logger.info(f"ZED camera opened: {self.resolution} @ {self.fps}fps")
            return True
            
        except ImportError:
            logger.error("pyzed not installed. Cannot use ZED camera.")
            return False
        except Exception as e:
            logger.error(f"Error opening ZED camera: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the ZED camera.
        
        Returns:
            BGR image as numpy array, or None if capture failed
        """
        if not self._is_open or self._camera is None:
            logger.warning("ZED camera not open")
            return None
            
        try:
            import pyzed.sl as sl
            
            # Grab a frame
            if self._camera.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
                return None
            
            # Retrieve left image (BGR format)
            self._camera.retrieve_image(self._image, sl.VIEW.LEFT)
            
            # Convert to numpy array
            frame = self._image.get_data()
            
            # ZED returns BGRA, convert to BGR
            if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def close(self) -> None:
        """Close the ZED camera."""
        if self._camera is not None:
            self._camera.close()
            self._is_open = False
            logger.info("ZED camera closed")
    
    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# =============================================================================
# Detection Classes
# =============================================================================


@dataclass
class Detection:
    """
    Represents a single person detection.

    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] in pixels
        confidence: Detection confidence score (0.0 to 1.0)
        center_x: X coordinate of bounding box center
        center_y: Y coordinate of bounding box center
        area: Area of bounding box in pixels squared
    """
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    center_x: float
    center_y: float
    area: float

    @classmethod
    def from_bbox(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float
    ) -> "Detection":
        """
        Create a Detection from bounding box coordinates.

        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            confidence: Detection confidence score

        Returns:
            Detection object with computed center and area
        """
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        area = width * height

        return cls(
            bbox=(x1, y1, x2, y2),
            confidence=confidence,
            center_x=center_x,
            center_y=center_y,
            area=area
        )


class PersonDetector:
    """
    YOLO-based person detector for visual servoing.

    This class wraps the Ultralytics YOLOv8 model and provides
    a simple interface for detecting persons in images.

    Usage:
        detector = PersonDetector(model_path="yolov8n.pt")
        detection = detector.detect(image)
        if detection is not None:
            print(f"Person found at {detection.center_x}, {detection.center_y}")

    Attributes:
        model_path: Path to YOLO model weights
        confidence_threshold: Minimum confidence to accept detection
        person_class_id: COCO class ID for person (always 0)
        use_gpu: Whether to use GPU acceleration
    """

    # COCO dataset class ID for "person"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.3,
        use_gpu: bool = True
    ) -> None:
        """
        Initialize the person detector.

        Args:
            model_path: Path to YOLO model weights file.
                        Options: yolov8n.pt (nano/fast), yolov8s.pt (small),
                                yolov8m.pt (medium), yolov8l.pt (large)
            confidence_threshold: Minimum confidence score to accept a detection
            use_gpu: If True, use GPU for inference (requires CUDA)

        Note:
            First initialization may download the model weights (~6MB for nano).
            Subsequent runs use the cached weights.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu

        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_path}")
        self._model = YOLO(model_path)

        # Set device (GPU or CPU)
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("CUDA not available, falling back to CPU")
                    self._device = "cpu"
            except ImportError:
                logger.warning("PyTorch CUDA not available, using CPU")
                self._device = "cpu"
        else:
            self._device = "cpu"
            logger.info("Using CPU for inference")

        # Statistics tracking
        self._frame_count = 0
        self._detection_count = 0

        logger.info(f"PersonDetector initialized (device: {self._device})")

    def detect(self, image: np.ndarray) -> Optional[Detection]:
        """
        Detect persons in an image and return the best detection.

        The "best" detection is the one with the largest bounding box area.
        This heuristic assumes the person to follow is closest to the camera.

        Args:
            image: Input image as numpy array (BGR format, as from OpenCV)

        Returns:
            Detection object if a person is found, None otherwise

        Processing Steps:
            1. Validate and convert image to proper format
            2. Run YOLO inference on the image
            3. Filter detections to only persons (class 0)
            4. Filter by confidence threshold
            5. Select the largest detection (closest person)
            6. Return Detection object with bbox, confidence, center, and area
        """
        if image is None:
            logger.warning("None image provided to detector")
            return None
        
        # Use duck typing to check for array-like properties
        # This avoids issues with numpy version mismatches in Docker
        if not (hasattr(image, 'shape') and hasattr(image, 'dtype') and hasattr(image, 'size')):
            logger.warning(f"Image does not have array-like properties, got {type(image)}")
            return None
        
        if image.size == 0:
            logger.warning("Empty image provided to detector")
            return None

        self._frame_count += 1

        try:
            # Ensure image is in proper format for YOLO
            # YOLO expects: numpy array, uint8, BGR or RGB, 3 channels
            processed_image = image
            
            # Convert to uint8 if needed
            if processed_image.dtype != np.uint8:
                if processed_image.max() <= 1.0:
                    processed_image = (processed_image * 255).astype(np.uint8)
                else:
                    processed_image = processed_image.astype(np.uint8)
            
            # Handle grayscale images (convert to BGR)
            if len(processed_image.shape) == 2:
                import cv2
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            elif len(processed_image.shape) == 3:
                # Ensure 3 channels (handle 4-channel RGBA/BGRA)
                if processed_image.shape[2] == 4:
                    import cv2
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2BGR)
                elif processed_image.shape[2] == 1:
                    import cv2
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            
            # Ensure contiguous array
            if not processed_image.flags['C_CONTIGUOUS']:
                processed_image = np.ascontiguousarray(processed_image)
            
            # Run YOLO inference
            # verbose=False suppresses per-frame logging
            results = self._model(
                processed_image,
                device=self._device,
                verbose=False,
                classes=[self.PERSON_CLASS_ID],  # Only detect persons
                conf=self.confidence_threshold
                
            )

            # Get detections from results
            # results[0].boxes contains all detections for this image
            if not results or len(results) == 0:
                return None

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return None

            # Extract person detections that meet confidence threshold
            person_detections: List[Detection] = []

            for i in range(len(boxes)):
                # Get confidence score
                conf = float(boxes.conf[i])

                # Check confidence threshold
                if conf < self.confidence_threshold:
                    continue

                # Get class ID (should be 0 for person, but verify)
                class_id = int(boxes.cls[i])
                if class_id != self.PERSON_CLASS_ID:
                    continue

                # Get bounding box coordinates
                # boxes.xyxy gives [x1, y1, x2, y2] format
                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox

                # Create Detection object
                detection = Detection.from_bbox(x1, y1, x2, y2, conf)
                person_detections.append(detection)

            # No valid detections found
            if not person_detections:
                return None

            # Select the largest detection (closest person)
            # This is a simple heuristic - the person we want to follow
            # is likely the closest one, which has the largest bbox
            best_detection = max(person_detections, key=lambda d: d.area)

            self._detection_count += 1

            logger.debug(
                f"Detection: center=({best_detection.center_x:.1f}, "
                f"{best_detection.center_y:.1f}), area={best_detection.area:.0f}, "
                f"conf={best_detection.confidence:.2f}"
            )

            return best_detection

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return None

    def detect_all(self, image: np.ndarray) -> List[Detection]:
        """
        Detect all persons in an image.

        Unlike detect(), this returns ALL person detections above
        the confidence threshold, not just the largest one.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of Detection objects, sorted by area (largest first)
        """
        if image is None or image.size == 0:
            return []

        try:
            results = self._model(
                image,
                device=self._device,
                verbose=False,
                classes=[self.PERSON_CLASS_ID]
            )

            if not results or len(results) == 0:
                return []

            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return []

            detections: List[Detection] = []

            for i in range(len(boxes)):
                conf = float(boxes.conf[i])

                if conf < self.confidence_threshold:
                    continue

                class_id = int(boxes.cls[i])
                if class_id != self.PERSON_CLASS_ID:
                    continue

                bbox = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox

                detection = Detection.from_bbox(x1, y1, x2, y2, conf)
                detections.append(detection)

            # Sort by area (largest first)
            detections.sort(key=lambda d: d.area, reverse=True)

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []

    def get_stats(self) -> dict:
        """
        Get detection statistics.

        Returns:
            Dictionary with frame count and detection rate
        """
        detection_rate = (
            self._detection_count / self._frame_count
            if self._frame_count > 0 else 0.0
        )

        return {
            "frame_count": self._frame_count,
            "detection_count": self._detection_count,
            "detection_rate": detection_rate,
            "device": self._device
        }

    def reset_stats(self) -> None:
        """Reset detection statistics counters."""
        self._frame_count = 0
        self._detection_count = 0


def main():
    """
    Test the PersonDetector with a sample image or webcam.

    This demonstrates basic detector functionality without
    requiring a connection to the Spot robot.
    """
    import cv2

    print("PersonDetector Test")
    print("=" * 40)

    # Initialize detector
    detector = PersonDetector(
        model_path="yolov8s.pt",
        confidence_threshold=0.3,
        use_gpu=True
    )

    # Try to open webcam for live testing
    print("Opening webcam for live detection...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam. Testing with a blank image.")
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        detection = detector.detect(test_image)
        print(f"Detection on blank image: {detection}")
        return

    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            detection = detector.detect(frame)

            # Draw detection if found
            if detection is not None:
                x1, y1, x2, y2 = [int(v) for v in detection.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"Person: {detection.confidence:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

                # Draw center point
                cx, cy = int(detection.center_x), int(detection.center_y)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Display frame
            cv2.imshow("Person Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        stats = detector.get_stats()
        print(f"\nStatistics:")
        print(f"  Frames processed: {stats['frame_count']}")
        print(f"  Detections: {stats['detection_count']}")
        print(f"  Detection rate: {stats['detection_rate']:.2%}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Detection Test Script
=====================
This script tests the YOLO person detection module.
It can run on saved images or live webcam feed.

Use this to verify:
- YOLO model loading and inference
- GPU acceleration (if available)
- Detection quality and performance
- Confidence thresholds

Usage:
    python tests/test_detection.py                  # Test with webcam
    python tests/test_detection.py --image test.jpg # Test on image file
    python tests/test_detection.py --benchmark      # Run performance benchmark

Author: Generated for Spot Person Follower Project
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perception import PersonDetector, Detection


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test YOLO person detection"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file to test on (if not specified, uses webcam)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLO model to use (default: yolov8s.pt)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )

    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Save detection result to file"
    )

    return parser.parse_args()


def draw_detection(
    image: np.ndarray,
    detection: Optional[Detection],
    inference_time: float = 0.0
) -> np.ndarray:
    """
    Draw detection result on image.

    Args:
        image: Input image
        detection: Detection result (or None)
        inference_time: Time taken for inference

    Returns:
        Annotated image
    """
    output = image.copy()
    h, w = output.shape[:2]

    # Draw center crosshair
    cx, cy = w // 2, h // 2
    cv2.line(output, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
    cv2.line(output, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

    if detection is not None:
        # Draw bounding box
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw center point
        det_cx = int(detection.center_x)
        det_cy = int(detection.center_y)
        cv2.circle(output, (det_cx, det_cy), 5, (0, 0, 255), -1)

        # Draw line to image center
        cv2.line(output, (cx, cy), (det_cx, det_cy), (0, 255, 255), 1)

        # Draw info
        info_lines = [
            f"Confidence: {detection.confidence:.2f}",
            f"Center: ({det_cx}, {det_cy})",
            f"Area: {detection.area:.0f} px^2",
            f"Inference: {inference_time*1000:.1f} ms"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(
                output, line,
                (x1, y2 + 20 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1
            )

        # Status
        cv2.putText(
            output, "PERSON DETECTED",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 255, 0), 2
        )

    else:
        # No detection
        cv2.putText(
            output, "NO DETECTION",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2
        )

        cv2.putText(
            output, f"Inference: {inference_time*1000:.1f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1
        )

    return output


def test_on_image(
    detector: PersonDetector,
    image_path: str,
    save_path: Optional[str] = None
) -> bool:
    """
    Test detection on a single image.

    Args:
        detector: PersonDetector instance
        image_path: Path to input image
        save_path: Path to save output (optional)

    Returns:
        True if successful
    """
    print(f"\nLoading image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return False

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")

    # Run detection
    print("Running detection...")
    start_time = time.time()
    detection = detector.detect(image)
    inference_time = time.time() - start_time

    # Print results
    if detection is not None:
        print(f"\n✓ Person detected:")
        print(f"  Confidence: {detection.confidence:.3f}")
        print(f"  Bounding box: {detection.bbox}")
        print(f"  Center: ({detection.center_x:.1f}, {detection.center_y:.1f})")
        print(f"  Area: {detection.area:.1f} pixels^2")
    else:
        print("\n✗ No person detected")

    print(f"\nInference time: {inference_time*1000:.1f} ms")

    # Draw result
    output = draw_detection(image, detection, inference_time)

    # Save or display
    if save_path:
        cv2.imwrite(save_path, output)
        print(f"Saved output to: {save_path}")
    else:
        print("Press any key to close...")
        cv2.imshow("Detection Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def test_on_webcam(detector: PersonDetector) -> None:
    """
    Test detection on live webcam feed.

    Args:
        detector: PersonDetector instance
    """
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    print("Webcam opened successfully")
    print("Press 'q' to quit, 's' to save frame")

    frame_count = 0
    fps_start = time.time()
    current_fps = 0.0
    debug_printed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        # Debug: Print frame info once
        if not debug_printed:
            try:
                print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, type: {type(frame)}")
            except Exception as e:
                print(f"Frame info error: {e}, type: {type(frame)}")
            debug_printed = True

        # Run detection
        start_time = time.time()
        detection = detector.detect(frame)
        inference_time = time.time() - start_time

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Draw result
        output = draw_detection(frame, detection, inference_time)

        # Add FPS
        cv2.putText(
            output, f"FPS: {current_fps:.1f}",
            (output.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2
        )

        cv2.imshow("Detection Test - Webcam", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, output)
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    # Print stats
    stats = detector.get_stats()
    print(f"\nStatistics:")
    print(f"  Frames processed: {stats['frame_count']}")
    print(f"  Detection rate: {stats['detection_rate']:.1%}")


def run_benchmark(detector: PersonDetector, num_frames: int = 100) -> None:
    """
    Run performance benchmark.

    Args:
        detector: PersonDetector instance
        num_frames: Number of frames to process
    """
    print(f"\nRunning benchmark ({num_frames} frames)...")

    # Create test image (640x480)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warm-up
    print("Warming up...")
    for _ in range(10):
        detector.detect(test_image)

    # Benchmark
    print("Benchmarking...")
    times = []

    for i in range(num_frames):
        start = time.time()
        detector.detect(test_image)
        times.append(time.time() - start)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames")

    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    avg_fps = 1.0 / np.mean(times)

    print(f"\n{'='*40}")
    print("Benchmark Results")
    print(f"{'='*40}")
    print(f"  Device: {detector._device}")
    print(f"  Model: {detector.model_path}")
    print(f"  Image size: 640 x 480")
    print(f"  Frames: {num_frames}")
    print(f"\n  Inference time:")
    print(f"    Average: {avg_time:.2f} ms")
    print(f"    Std dev: {std_time:.2f} ms")
    print(f"    Min: {min_time:.2f} ms")
    print(f"    Max: {max_time:.2f} ms")
    print(f"\n  Throughput: {avg_fps:.1f} FPS")
    print(f"{'='*40}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("YOLO Person Detection Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"GPU: {'Disabled' if args.no_gpu else 'Enabled (if available)'}")
    print("=" * 60)

    # Initialize detector
    print("\nInitializing detector...")
    detector = PersonDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        use_gpu=not args.no_gpu
    )

    stats = detector.get_stats()
    print(f"Detector ready (device: {stats['device']})")

    try:
        if args.benchmark:
            # Run benchmark
            run_benchmark(detector)

        elif args.image:
            # Test on image file
            test_on_image(
                detector,
                args.image,
                save_path=args.save_output
            )

        else:
            # Test on webcam
            test_on_webcam(detector)

        print("\n✓ Detection test completed successfully")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

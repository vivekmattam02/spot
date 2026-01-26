# =============================================================================
# Spot Person Follower - Dockerfile (with ZED SDK)
# =============================================================================
# This Dockerfile creates a container with all dependencies for running
# the Spot person-following system with ZED 2i camera support.
#
# BUILD:
#   docker build -t spot-person-follower .
#
# RUN (with GPU + ZED camera):
#   docker run --gpus all --privileged --network host -it spot-person-follower
#
# RUN (with GPU + Display + ZED camera):
#   xhost +local:docker
#   docker run --gpus all --privileged --network host -e DISPLAY=$DISPLAY \
#              -v /tmp/.X11-unix:/tmp/.X11-unix -it spot-person-follower
#
# =============================================================================

# -----------------------------------------------------------------------------
# Base image: Stereolabs ZED SDK with CUDA 12.1 and Ubuntu 22.04
# This includes: CUDA runtime, cuDNN, ZED SDK, pyzed Python package
# -----------------------------------------------------------------------------
FROM stereolabs/zed:4.2-runtime-cuda12.1-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python to not buffer output (useful for logging)
ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Install system dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.10
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # X11 for visualization (optional)
    libx11-6 \
    libxcb1 \
    # Networking tools (useful for debugging)
    iputils-ping \
    net-tools \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# -----------------------------------------------------------------------------
# Set working directory
# -----------------------------------------------------------------------------
WORKDIR /app

# -----------------------------------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------------------------------

# IMPORTANT: Base ZED image has pyzed compiled against numpy 1.26.4.
# Other packages may upgrade numpy which breaks pyzed.
# We force reinstall numpy 1.26.4 at the end to restore compatibility.

# Install PyTorch with CUDA support (separate layer for caching)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install ultralytics (YOLO)
RUN pip install ultralytics

# Install Boston Dynamics SDK
RUN pip install bosdyn-client bosdyn-api

# Install opencv-headless (no display needed)
RUN pip install opencv-python-headless

# Install Flask for web-based camera streaming
RUN pip install flask

# CRITICAL: Base ZED image has numpy 1.26.4 with pyzed compiled against it.
# PyTorch/ultralytics install numpy 2.x which breaks pyzed binary compatibility.
# Force reinstall numpy 1.26.4 AFTER all other packages to restore compatibility.
RUN pip install --force-reinstall numpy==1.26.4

# Download YOLO model weights (so it's cached in the image)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# -----------------------------------------------------------------------------
# Copy application code
# -----------------------------------------------------------------------------
COPY config/ ./config/
COPY src/ ./src/
COPY utils/ ./utils/
COPY tests/ ./tests/

# -----------------------------------------------------------------------------
# Verify installation (NOTE: ZED SDK verified at runtime, not build time)
# -----------------------------------------------------------------------------
RUN echo "=== Verifying Installation ===" \
    && python --version \
    && python -c "import torch; print(f'PyTorch: {torch.__version__}')" \
    && python -c "import bosdyn.client; print('bosdyn-client: OK')" \
    && python -c "from ultralytics import YOLO; print('ultralytics: OK')" \
    && python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" \
    && echo "=== All dependencies OK (ZED SDK verified at runtime) ==="

# -----------------------------------------------------------------------------
# Default command
# -----------------------------------------------------------------------------
# Default: run the main person follower
CMD ["python", "src/main.py"]

#!/bin/bash
# =============================================================================
# Spot Person Follower - Environment Setup Script
# =============================================================================
# This script creates a Python virtual environment and installs dependencies.
#
# REQUIREMENTS:
#   - Python 3.8, 3.9, or 3.10 (NOT 3.11 or 3.12)
#   - pip
#   - virtualenv (will be installed if missing)
#
# USAGE:
#   chmod +x setup_environment.sh
#   ./setup_environment.sh
#
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "Spot Person Follower - Environment Setup"
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Check Python version
# -----------------------------------------------------------------------------
echo "[1/6] Checking Python version..."

# Try to find a compatible Python
PYTHON_CMD=""

for cmd in python3.10 python3.9 python3.8 python3 python; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | grep -oP '\d+\.\d+')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)

        if [ "$major" = "3" ] && [ "$minor" -ge 8 ] && [ "$minor" -le 10 ]; then
            PYTHON_CMD=$cmd
            echo "    Found compatible Python: $cmd (version $version)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "ERROR: No compatible Python found!"
    echo ""
    echo "This project requires Python 3.8, 3.9, or 3.10"
    echo ""
    echo "Your installed Python versions:"
    which python python3 python3.8 python3.9 python3.10 2>/dev/null || echo "    None found in PATH"
    echo ""
    echo "To install Python 3.10 on Ubuntu:"
    echo "    sudo apt update"
    echo "    sudo apt install python3.10 python3.10-venv python3.10-dev"
    echo ""
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo "    Using: $PYTHON_VERSION"
echo ""

# -----------------------------------------------------------------------------
# Create virtual environment
# -----------------------------------------------------------------------------
echo "[2/6] Creating virtual environment..."

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "    Virtual environment already exists at ./$VENV_DIR"
    read -p "    Delete and recreate? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        rm -rf "$VENV_DIR"
        echo "    Deleted old environment"
    else
        echo "    Keeping existing environment"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "    Created virtual environment at ./$VENV_DIR"
fi
echo ""

# -----------------------------------------------------------------------------
# Activate virtual environment
# -----------------------------------------------------------------------------
echo "[3/6] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "    Activated: $VIRTUAL_ENV"
echo ""

# -----------------------------------------------------------------------------
# Upgrade pip
# -----------------------------------------------------------------------------
echo "[4/6] Upgrading pip..."
pip install --upgrade pip
echo ""

# -----------------------------------------------------------------------------
# Install PyTorch with CUDA support
# -----------------------------------------------------------------------------
echo "[5/6] Installing PyTorch with CUDA support..."
echo "    (This may take a few minutes)"
echo ""

# Detect CUDA version
CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "    Detected CUDA version: $CUDA_VERSION"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "    NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/        /'
fi

# Install PyTorch based on CUDA availability
if [ -n "$CUDA_VERSION" ]; then
    # Parse CUDA version
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "    Installing PyTorch for CUDA 12.x..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        echo "    Installing PyTorch for CUDA 11.8..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "    CUDA version may be outdated. Installing default PyTorch..."
        pip install torch torchvision
    fi
else
    echo "    No CUDA detected. Installing CPU-only PyTorch..."
    echo "    (Detection will be slower without GPU)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
echo ""

# -----------------------------------------------------------------------------
# Install remaining dependencies
# -----------------------------------------------------------------------------
echo "[6/6] Installing remaining dependencies..."
pip install -r requirements.txt --ignore-installed torch torchvision
echo ""

# -----------------------------------------------------------------------------
# Verify installation
# -----------------------------------------------------------------------------
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"
echo ""

echo "Python version:"
python --version
echo ""

echo "PyTorch:"
python -c "import torch; print(f'    Version: {torch.__version__}'); print(f'    CUDA available: {torch.cuda.is_available()}'); print(f'    CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

echo "Boston Dynamics SDK:"
python -c "import bosdyn.client; print(f'    bosdyn-client version: {bosdyn.client.__version__ if hasattr(bosdyn.client, \"__version__\") else \"installed\"}')" 2>/dev/null || echo "    WARNING: bosdyn-client import failed"
echo ""

echo "YOLO:"
python -c "from ultralytics import YOLO; print('    ultralytics installed successfully')"
echo ""

echo "OpenCV:"
python -c "import cv2; print(f'    Version: {cv2.__version__}')"
echo ""

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment in the future:"
echo "    source venv/bin/activate"
echo ""
echo "To run the person follower:"
echo "    source venv/bin/activate"
echo "    python src/main.py"
echo ""
echo "To deactivate when done:"
echo "    deactivate"
echo ""

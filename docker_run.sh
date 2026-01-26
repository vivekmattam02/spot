#!/bin/bash
# =============================================================================
# Spot Person Follower - Docker Helper Script
# =============================================================================
#
# USAGE:
#   ./docker_run.sh build          - Build the Docker image
#   ./docker_run.sh run            - Run person follower
#   ./docker_run.sh run-display    - Run with visualization
#   ./docker_run.sh test-detection - Test YOLO detection
#   ./docker_run.sh test-camera    - Test Spot camera
#   ./docker_run.sh test-mobility  - Test robot movement
#   ./docker_run.sh shell          - Open interactive shell
#   ./docker_run.sh check          - Check prerequisites
#
# =============================================================================

set -e

IMAGE_NAME="spot-person-follower"
CONTAINER_NAME="spot-follower"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

check_prerequisites() {
    echo "============================================================"
    echo "Checking Prerequisites"
    echo "============================================================"
    echo ""

    # Check Docker
    echo -n "Docker: "
    if command -v docker &> /dev/null; then
        docker_version=$(docker --version)
        echo -e "${GREEN}$docker_version${NC}"
    else
        echo -e "${RED}NOT INSTALLED${NC}"
        echo "  Install: https://docs.docker.com/engine/install/"
        exit 1
    fi

    # Check Docker Compose
    echo -n "Docker Compose: "
    if command -v docker-compose &> /dev/null; then
        compose_version=$(docker-compose --version)
        echo -e "${GREEN}$compose_version${NC}"
    elif docker compose version &> /dev/null; then
        compose_version=$(docker compose version)
        echo -e "${GREEN}$compose_version (plugin)${NC}"
    else
        echo -e "${RED}NOT INSTALLED${NC}"
        exit 1
    fi

    # Check NVIDIA Docker
    echo -n "NVIDIA Container Toolkit: "
    if command -v nvidia-container-cli &> /dev/null; then
        echo -e "${GREEN}Installed${NC}"
    elif docker info 2>/dev/null | grep -q "nvidia"; then
        echo -e "${GREEN}Available${NC}"
    else
        echo -e "${YELLOW}NOT DETECTED${NC}"
        echo "  GPU support may not work!"
        echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi

    # Check NVIDIA GPU
    echo -n "NVIDIA GPU: "
    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo -e "${GREEN}$gpu_name${NC}"
    else
        echo -e "${YELLOW}nvidia-smi not found${NC}"
    fi

    # Check network to Spot
    echo -n "Network to Spot (192.168.80.3): "
    if ping -c 1 -W 2 192.168.80.3 &> /dev/null; then
        echo -e "${GREEN}Reachable${NC}"
    else
        echo -e "${YELLOW}Not reachable (is Spot on?)${NC}"
    fi

    echo ""
    echo "============================================================"
}

build_image() {
    echo "============================================================"
    echo "Building Docker Image: $IMAGE_NAME"
    echo "============================================================"
    echo ""

    docker build -t $IMAGE_NAME:latest .

    echo ""
    echo -e "${GREEN}Build complete!${NC}"
}

run_follower() {
    echo "Running person follower..."

    # Stop existing container if running
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    docker run \
        --gpus all \
        --network host \
        --name $CONTAINER_NAME \
        -v "$(pwd)/config:/app/config:ro" \
        -it \
        $IMAGE_NAME:latest \
        python src/main.py
}

run_follower_display() {
    echo "Running person follower with display..."

    # Allow X11 access
    xhost +local:docker 2>/dev/null || echo "Warning: xhost command failed"

    # Stop existing container if running
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    docker run \
        --gpus all \
        --network host \
        --name $CONTAINER_NAME \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$(pwd)/config:/app/config:ro" \
        -it \
        $IMAGE_NAME:latest \
        python src/main.py
}

test_detection() {
    echo "Testing YOLO detection with webcam..."

    xhost +local:docker 2>/dev/null || true

    docker run \
        --gpus all \
        --network host \
        --rm \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        --device /dev/video0:/dev/video0 \
        -it \
        $IMAGE_NAME:latest \
        python tests/test_detection.py
}

test_camera() {
    echo "Testing Spot camera..."

    mkdir -p test_outputs

    docker run \
        --gpus all \
        --network host \
        --rm \
        -v "$(pwd)/config:/app/config:ro" \
        -v "$(pwd)/test_outputs:/app/test_outputs" \
        -it \
        $IMAGE_NAME:latest \
        python tests/test_camera.py --save-path /app/test_outputs/camera_test.jpg

    echo ""
    if [ -f "test_outputs/camera_test.jpg" ]; then
        echo -e "${GREEN}Image saved to test_outputs/camera_test.jpg${NC}"
    fi
}

test_mobility() {
    echo "============================================================"
    echo -e "${YELLOW}WARNING: This will make the robot MOVE!${NC}"
    echo "============================================================"
    echo ""

    docker run \
        --gpus all \
        --network host \
        --rm \
        -v "$(pwd)/config:/app/config:ro" \
        -it \
        $IMAGE_NAME:latest \
        python tests/test_mobility.py
}

run_shell() {
    echo "Opening interactive shell..."

    xhost +local:docker 2>/dev/null || true

    docker run \
        --gpus all \
        --network host \
        --rm \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$(pwd)/config:/app/config:ro" \
        -v "$(pwd)/src:/app/src:ro" \
        -v "$(pwd)/tests:/app/tests:ro" \
        -it \
        $IMAGE_NAME:latest \
        /bin/bash
}

show_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  build          - Build the Docker image"
    echo "  run            - Run person follower (no display)"
    echo "  run-display    - Run person follower with visualization"
    echo "  test-detection - Test YOLO detection with webcam"
    echo "  test-camera    - Test Spot camera connection"
    echo "  test-mobility  - Test robot movement (ROBOT WILL MOVE!)"
    echo "  shell          - Open interactive shell in container"
    echo "  check          - Check prerequisites"
    echo ""
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

case "$1" in
    build)
        build_image
        ;;
    run)
        run_follower
        ;;
    run-display)
        run_follower_display
        ;;
    test-detection)
        test_detection
        ;;
    test-camera)
        test_camera
        ;;
    test-mobility)
        test_mobility
        ;;
    shell)
        run_shell
        ;;
    check)
        check_prerequisites
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

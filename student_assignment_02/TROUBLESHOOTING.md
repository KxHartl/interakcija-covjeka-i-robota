# Troubleshooting Guide - Nose Detection Program

## Common Issues and Solutions

### 1. Camera Access Issues

#### Error: "Error: Cannot access camera"

**Causes:**
- Camera not connected properly
- Another program using camera
- Insufficient permissions (Linux/Mac)

**Solutions:**

```bash
# Linux - Check permissions:
ls -l /dev/video0
chmod 666 /dev/video0

# macOS - Allow camera access in settings
# Windows - Check Device Manager
```

**Try different camera index:**

```python
cap = cv2.VideoCapture(1)  # Try second camera
```

### 2. Model Loading Issues

#### Error: "Error loading model"

**Solution 1: Auto download model**

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"
```

### 3. Missing Dependencies

#### Error: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:**

```bash
pip install -r src/requirements.txt
```

### 4. Performance Issues

#### Problem: "Slow detection"

**Solution 1: Use faster model**

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

**Solution 2: Lower camera resolution**

```python
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Instead of 1280
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Instead of 720
```

**Solution 3: Install GPU driver (CUDA)**

For NVIDIA cards, install CUDA and cuDNN for GPU acceleration.

### 5. No Face Detected

#### Problem: "No face detected"

**Causes:**
- Poor lighting
- Dirty camera lens
- Face too small in frame
- Facing wrong direction

**Solutions:**

```python
# Lower confidence threshold
detector = NoseDetector(confidence_threshold=0.3)

# Improve lighting
# Clean camera lens
# Get closer to camera
```

### 6. Keyboard/Input Issues

#### Problem: "'q' doesn't stop program"

**Solution:**
- Click on OpenCV window first (focus it)
- Try ESC key
- Press Ctrl+C in terminal

### 7. Memory Issues

#### Error: "MemoryError" or program hangs

**Solution: Reduce resolution and use faster model**

```python
detector = NoseDetector(model_path="yolo11n-pose.pt")
```

## Quick Diagnostics

```bash
# Check Python version
python --version

# Check OpenCV
python -c "import cv2; print(cv2.__version__)"

# Check YOLO
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Check camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED')"
```

## Hardware Requirements

### Minimum
- CPU: Dual-core 2GHz
- RAM: 4GB
- Camera: HD (720p)

### Recommended
- CPU: Quad-core 2.5GHz+
- RAM: 8GB+
- GPU: NVIDIA (CUDA capable)
- Camera: Full HD (1080p)

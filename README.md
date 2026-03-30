# Cerebral Palsy Comprehension ML

**Machine learning system for gesture-based communication assistance for individuals with Cerebral Palsy.**

This project explores using computer vision and deep learning to recognize hand gestures and facial mouth movements, enabling people with motor speech disorders to communicate through physical gestures. The system maps recognized gestures to actions or communication intents in real time.

## Overview

Cerebral Palsy (CP) affects movement and motor control, making speech difficult or impossible for many individuals. This project builds two complementary recognition pipelines:

1. **Hand Gesture Recognition** — A CNN trained to classify hand gestures from webcam input, allowing hands-free control and communication
2. **Mouth/Facial Landmark Detection** — Detects mouth shape and movements using facial landmark tracking to supplement gesture input

The gesture recognition model achieves real-time classification at 30 FPS with configurable confidence thresholds, and can optionally map gestures to system controls (volume, media playback).

## Features

- Real-time gesture classification using a trained CNN (TensorFlow/Keras)
- Background subtraction and frame preprocessing for robust hand isolation
- 6-gesture vocabulary: Left hand up, Right hand up, Both hands up, Hands down, Victory sign, Thumbs up
- Configurable confidence threshold to reduce false positives
- Optional system control mode (map gestures to keyboard/volume actions)
- Gesture recording — save recognition sessions as timestamped JSON logs
- Debug mode with live foreground mask visualization
- Mouth detection module using Dlib facial landmarks + Haar cascades
- Training pipeline with data augmentation, early stopping, and learning rate scheduling

## Model Architecture

The gesture classifier is a **CNN with three convolutional blocks**:
- Conv2D → BatchNorm → ReLU (×2) → MaxPool → Dropout
- Repeated 3 times with 32 → 64 → 128 filters
- Global Average Pooling → Dense(512) → Softmax output

Trained with data augmentation (rotation, shifts, zoom, flip) and Adam optimizer. Training outputs include accuracy/loss plots, a confusion matrix, and per-epoch history CSV.

## Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** — model training and inference
- **OpenCV** — real-time video capture, background subtraction, image processing
- **Dlib** — facial landmark detection (68-point predictor)
- **NumPy, scikit-learn** — data preprocessing and evaluation
- **Matplotlib, Seaborn** — training visualization
- **pyautogui** — gesture-to-system-control mapping

## Setup

### Prerequisites

- Python 3.8+
- Webcam
- Dlib shape predictor model (`shape_predictor_68_face_landmarks.dat`) — download from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

### Installation

```bash
git clone https://github.com/RedNetty/CerebralPalsyComprehensionML.git
cd CerebralPalsyComprehensionML
pip install tensorflow opencv-python dlib numpy scikit-learn matplotlib seaborn pyautogui pandas
```

### Model Weights

**Pre-trained model weights are not included in this repository** (binary `.h5` files are excluded from version control due to size). To get started, you must either:

- **Train from scratch** using the steps below, or
- **Download pre-trained weights** (if made available via GitHub Releases) and place them at the paths expected by the scripts (`Hand_Gesture_Recognize.h5` and `Updated Model/Gesture_Model.h5`)

### Training the Model

1. Organize your gesture images into folders by class number:
   ```
   gesture/
   ├── 0/    # Left hand up images
   ├── 1/    # Hands down images
   ├── 2/    # Right hand up images
   ...
   ```

2. Run the training script:
   ```bash
   python Gesture_Train.py
   ```

Training outputs (model checkpoints, plots, confusion matrix, history CSV) are saved to a timestamped `training_output_*/` directory.

### Running Real-Time Recognition

```bash
python __main__.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Toggle mute |
| `r` | Start/stop recording session |
| `d` | Toggle debug mode (show foreground masks) |
| `s` | Toggle system control mode |
| `+` / `-` | Increase/decrease confidence threshold |

### Mouth Detection Module

```bash
cd mouth_recognition
python Mouth_Detection.py
```

Requires `shape_predictor_68_face_landmarks.dat` in the `mouth_recognition/` directory.

## Project Structure

```
├── __main__.py                    # Real-time gesture recognition entry point
├── Gesture_Train.py               # CNN training pipeline
├── Image_Capture_Static.py        # Static image capture utility
├── Recognize_Gesture2.py          # Alternative recognition script
├── Hand_Gesture_Recognize.h5      # Pre-trained model weights (not tracked — train locally or download separately)
├── mouth_recognition/
│   ├── Mouth_Detection.py         # Facial landmark + mouth detection
│   ├── haarcascade_frontalface_alt2.xml
│   └── shape_predictor_68_face_landmarks.dat
└── Updated Model/
    ├── Gesture_Train.py           # Improved training pipeline
    ├── Process_Videos.py          # Video dataset processing
    ├── Recognize_Video.py         # Recognize gestures in video files
    └── Recognize_Video_Live.py    # Live video recognition
```

## Motivation

Standard assistive communication devices are expensive, device-specific, and require significant setup. This project explores a low-cost, camera-only approach that can run on consumer hardware and be personalized to an individual's specific movement patterns — making communication assistance more accessible.

## License

Open source. See repository for license details.

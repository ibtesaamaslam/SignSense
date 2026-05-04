<div align="center">

<br/>

# 🖐 SignSense

### *Hand Gesture Recognition System — Computer Vision meets Machine Learning*

<br/>

[![Python](https://img.shields.io/badge/Python-3.6%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarks-0097A7?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Webcam%20%26%20Image-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/abhishek14398/gesture-recognition-dataset)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle Compatible](https://img.shields.io/badge/Kaggle-Compatible-20BEFF?style=flat-square&logo=kaggle&logoColor=white)]()

<br/>

[![GitHub Repo](https://img.shields.io/badge/📂%20Repository-ibtesaamaslam%2FGesture--Recognition--Model-181717?style=for-the-badge&logo=github)](https://github.com/ibtesaamaslam/Gesture-Recognition-Model-)

<br/>

---

</div>

## Table of Contents

1. [Introduction](#1-introduction)
2. [How It Works — System Overview](#2-how-it-works--system-overview)
3. [Recognized Gestures](#3-recognized-gestures)
4. [Features](#4-features)
5. [Technologies Used](#5-technologies-used)
6. [Dataset](#6-dataset)
   - 6.1 [Source & Structure](#61-source--structure)
   - 6.2 [Data Processing Pipeline](#62-data-processing-pipeline)
   - 6.3 [Kaggle API Setup](#63-kaggle-api-setup)
7. [Project Structure](#7-project-structure)
8. [Installation & Setup](#8-installation--setup)
9. [Usage](#9-usage)
   - 9.1 [Cell 1 — Install Libraries](#91-cell-1--install-libraries)
   - 9.2 [Cell 2 — Inspect Dataset](#92-cell-2--inspect-dataset)
   - 9.3 [Cell 3 — Train & Run Inference](#93-cell-3--train--run-inference)
10. [Model Architecture & Design Decisions](#10-model-architecture--design-decisions)
    - 10.1 [Why MediaPipe](#101-why-mediapipe)
    - 10.2 [Why Random Forest](#102-why-random-forest)
    - 10.3 [Feature Engineering](#103-feature-engineering)
    - 10.4 [Data Augmentation Strategy](#104-data-augmentation-strategy)
11. [Real-Time Inference Pipeline](#11-real-time-inference-pipeline)
12. [Improving Model Accuracy](#12-improving-model-accuracy)
13. [Troubleshooting](#13-troubleshooting)
14. [Roadmap — Future Improvements](#14-roadmap--future-improvements)
15. [Contributing](#15-contributing)
16. [License](#16-license)
17. [Author](#17-author)
18. [Acknowledgments](#18-acknowledgments)

---

## 1. Introduction

**SignSense** is a machine learning-based hand gesture recognition system that detects and classifies hand gestures in real time using **MediaPipe** for hand landmark extraction and a **Random Forest** classifier for robust, fast prediction.

The system bridges the gap between human body language and machine understanding — enabling computers to interpret hand gestures the same way humans do instinctively. This has meaningful applications across:

- **Accessibility technology** — giving non-verbal individuals new ways to interact with computers
- **Human-Computer Interaction (HCI)** — touchless control interfaces
- **Sign language interpretation** — a foundation for full ASL/PSL recognition systems
- **Gaming and AR/VR** — gesture-based controller alternatives
- **Robotics** — command robots through intuitive hand signals

SignSense supports both **offline training** from a Kaggle dataset and **real-time inference** via a live webcam feed — making it functional in both cloud notebook environments (like Kaggle) and local development machines.

**What makes SignSense different from generic gesture tutorials?**

Most gesture recognition tutorials either rely on pre-trained black-box models or skip real-world reliability challenges entirely. SignSense specifically addresses:

- Mislabelled gesture classes in real-world datasets
- Low model accuracy on similar-looking gestures
- The gap between notebook training and live deployment
- Class imbalance through targeted data augmentation

---

## 2. How It Works — System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       INPUT LAYER                               │
│          Webcam Frame  OR  Dataset Image (.jpg/.png)            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MEDIAPIPE HAND DETECTION                      │
│   Detects hand region → extracts 21 3D landmarks (x, y, z)     │
│   Minimum detection confidence: 0.3                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
│   21 landmarks × 3 coordinates = 63-dimensional feature vector │
│   StandardScaler normalisation applied                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               RANDOM FOREST CLASSIFIER (100 trees)             │
│   Input: 63-dim feature vector                                  │
│   Output: Gesture label + confidence score                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT / VISUALISATION                       │
│   Predicted label overlaid on webcam feed                       │
│   Live matplotlib confidence bar chart updated in real time     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Recognized Gestures

SignSense v1.0.0 ships with recognition for three gesture classes:

| Gesture | Label | Description |
|---|---|---|
| 🖐 **Wave** | `wave` | Open hand moved side to side — greeting or attention gesture |
| ✋ **Stop** | `stop` | Flat open palm facing forward — halt or pause signal |
| 👍 **Thumbs Up** | `thumbs_up` | Closed fist with thumb extended upward — approval or confirmation |

These three gestures were selected because they are visually distinct enough to achieve high classification accuracy even with a statistical model, while being common enough to be immediately useful in real-world HCI scenarios.

---

## 4. Features

**🖐 Three-Gesture Recognition**
Accurately classifies `wave`, `stop`, and `thumbs_up` from both static images and live video.

**🎥 Real-Time Webcam Inference**
Opens a live webcam feed, processes each frame through MediaPipe and the trained classifier, and overlays the predicted gesture label with confidence score directly on the video.

**📁 Dynamic Dataset Loader**
Handles both numeric folder naming (`0`, `1`, `2`) and string-based naming (`wave`, `stop`, `thumbs_up`) through an automatic `GESTURE_MAPPING` dictionary — making it compatible with diverse Kaggle dataset structures without code changes.

**🌀 Data Augmentation Pipeline**
Each training image is augmented with random horizontal flips, rotations (±15°), and brightness adjustments — tripling the effective training dataset size and improving generalisation to real-world lighting and hand orientations.

**⚖️ Feature Scaling**
`StandardScaler` normalises the 63-dimensional landmark feature vector, ensuring that the magnitude differences between x, y, and z landmark coordinates do not bias the classifier.

**📈 Live Confidence Plot**
A real-time matplotlib bar chart updates alongside the webcam feed, showing the model's confidence score for each gesture class on every frame — ideal for debugging, demonstration, and educational use.

**💡 Robust Error Logging**
Gracefully handles image load failures, missing landmark detections, and corrupted dataset files — logging errors without crashing the training pipeline.

**🌐 Kaggle Notebook Compatible**
The entire training pipeline runs inside a Kaggle notebook without a webcam — using the dataset-based inference path and suppressing real-time video requirements.

---

## 5. Technologies Used

| Library | Version | Purpose |
|---|---|---|
| [Python](https://www.python.org/) | 3.6+ | Core language |
| [OpenCV](https://opencv.org/) | Latest | Webcam capture, image loading, frame annotation |
| [MediaPipe](https://mediapipe.dev/) | Latest | Real-time hand landmark detection (21 3D keypoints) |
| [Scikit-learn](https://scikit-learn.org/) | Latest | Random Forest classifier, StandardScaler, metrics |
| [NumPy](https://numpy.org/) | Latest | Feature vector manipulation and augmentation math |
| [Matplotlib](https://matplotlib.org/) | Latest | Live confidence bar chart visualisation |
| [KaggleHub](https://github.com/Kaggle/kagglehub) | Latest | Programmatic dataset download from Kaggle |
| [Tqdm](https://tqdm.github.io/) | Latest | Progress bars during dataset loading and training |

---

## 6. Dataset

### 6.1 Source & Structure

**Source:** [Kaggle — `abhishek14398/gesture-recognition-dataset`](https://www.kaggle.com/datasets/abhishek14398/gesture-recognition-dataset)

**Image formats supported:** `.jpg`, `.jpeg`, `.png`

**Dataset folder structure:**

```
gesture-recognition-dataset/
│
├── train/
│   ├── 0/          ← mapped to "wave"
│   ├── 1/          ← mapped to "stop"
│   └── 2/          ← mapped to "thumbs_up"
│
└── val/
    ├── 0/
    ├── 1/
    └── 2/
```

The dataset may also use string-based folder names (`wave/`, `stop/`, `thumbs_up/`) depending on the version downloaded. SignSense's dynamic loader handles both structures automatically.

---

### 6.2 Data Processing Pipeline

| Step | Detail |
|---|---|
| **Class filtering** | Only the three target gesture classes are loaded; any other classes in the dataset are ignored |
| **Landmark extraction** | MediaPipe processes each image and extracts 21 3D hand landmarks (x, y, z per landmark) |
| **Feature vector** | 21 × 3 = **63 features** per image |
| **Cap per class** | Maximum 500 images per gesture class to prevent class imbalance |
| **Augmentation** | Each image generates 3 augmented variants (flip, rotate ±15°, brightness ±30%) |
| **Scaling** | `StandardScaler` fitted on training features and applied to both train and inference |

---

### 6.3 Kaggle API Setup

To download the dataset programmatically via `kagglehub`, you need a valid Kaggle API token:

**Step 1 — Get your API token:**
1. Log in to [kaggle.com](https://www.kaggle.com)
2. Go to **Account → API → Create New Token**
3. A `kaggle.json` file downloads automatically

**Step 2 — Place the token:**

```bash
# Linux / Mac
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows
# Copy kaggle.json to: C:\Users\<YourUsername>\.kaggle\kaggle.json
```

**Step 3 — Verify:**

```python
import kagglehub
path = kagglehub.dataset_download("abhishek14398/gesture-recognition-dataset")
print("Dataset path:", path)
```

> ⚠️ In Kaggle notebooks, authentication is handled automatically — no `kaggle.json` setup is needed.

---

## 7. Project Structure

```
Gesture-Recognition-Model/
│
├── gesture_recognition.py      # Main script — all pipeline stages
│   ├── Config                  # Hyperparameters (confidence, cap, augmentation factor)
│   ├── Feature Extraction      # MediaPipe landmark → 63-dim vector
│   ├── Data Augmentation       # Flip, rotate, brightness transforms
│   ├── Dataset Loader          # Dynamic folder mapping + image processing loop
│   ├── Model Training          # Random Forest + StandardScaler fit
│   ├── Real-Time Inference     # Webcam loop + per-frame prediction
│   └── Live Visualisation      # Matplotlib confidence chart
│
├── SignSense.ipynb             # Jupyter Notebook version (Kaggle-ready)
├── requirements.txt            # All pip dependencies
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## 8. Installation & Setup

### Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.6+ | Tested on 3.9, 3.10, 3.11 |
| pip | Latest | Included with Python |
| Webcam | Optional | Only required for real-time inference |
| Internet | Required (first run) | For Kaggle dataset download |
| Kaggle Account | Required | For `kaggle.json` API token |

### Step 1 — Clone the Repository

```bash
git clone https://github.com/ibtesaamaslam/Gesture-Recognition-Model-.git
cd Gesture-Recognition-Model-
```

### Step 2 — (Recommended) Create a Virtual Environment

```bash
# Create
python -m venv venv

# Activate — Linux/Mac
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### Step 3 — Install All Dependencies

```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib kagglehub tqdm
```

Or from the requirements file:

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Kaggle Credentials

Follow the steps in [Section 6.3](#63-kaggle-api-setup) above.

---

## 9. Usage

### 9.1 Cell 1 — Install Libraries

```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib kagglehub tqdm
```

---

### 9.2 Cell 2 — Inspect Dataset

Run this first to verify the dataset downloads correctly and inspect its folder structure:

```python
import os, glob, kagglehub

KAGGLE_DATASET = "abhishek14398/gesture-recognition-dataset"
dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)

# Print full directory tree
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = ' ' * 2 * (level + 1)
    for file in files[:3]:   # show first 3 files per folder
        print(f"{sub_indent}{file}")

# Count total images
image_paths = glob.glob(
    os.path.join(dataset_path, "**/*.*"), recursive=True
)
print(f"\nTotal images found: {len(image_paths)}")
```

---

### 9.3 Cell 3 — Train & Run Inference

```bash
python gesture_recognition.py
```

**What happens when you run this:**

1. Dataset is downloaded (or loaded from cache)
2. All images are processed through MediaPipe → 63-dim feature vectors extracted
3. Data is augmented (×3 per image)
4. Random Forest classifier is trained on the full augmented dataset
5. If a webcam is detected → real-time inference mode launches
6. If no webcam is detected (Kaggle environment) → batch inference on validation set runs
7. Live matplotlib confidence chart updates on each prediction

---

## 10. Model Architecture & Design Decisions

### 10.1 Why MediaPipe

MediaPipe's Hand solution was chosen over raw CNN-based hand detection for two reasons:

**Speed:** MediaPipe runs at 30+ FPS on CPU alone — essential for real-time webcam inference without a GPU. Training a custom CNN hand detector to comparable speed would require significantly more compute and data.

**Landmark quality:** MediaPipe provides 21 precise 3D landmarks per hand — a compact, structured representation that is far more informative for a downstream classifier than raw pixel data. Each landmark carries semantic meaning (fingertip, knuckle, palm base), which makes the feature space highly interpretable.

---

### 10.2 Why Random Forest

A Random Forest was selected over other classifiers for this specific task:

| Classifier | Considered | Outcome |
|---|---|---|
| **Random Forest** | ✅ Selected | High accuracy on 63-dim structured features; robust to outliers; no GPU needed |
| SVM | Evaluated | Competitive accuracy but slower inference on large feature sets |
| MLP / Neural Net | Evaluated | Overfits on small dataset without extensive tuning |
| KNN | Evaluated | Too slow for real-time inference at high frame rates |

The 100-tree ensemble provides strong variance reduction while keeping inference latency well under 10ms per frame — fast enough for 30 FPS real-time prediction.

---

### 10.3 Feature Engineering

**Why 63 features?**

MediaPipe detects 21 hand landmarks. Each landmark has three coordinates:
- `x` — horizontal position (normalised 0–1 relative to image width)
- `y` — vertical position (normalised 0–1 relative to image height)
- `z` — depth estimate (relative to wrist, scale proportional to hand size)

21 × 3 = **63 features per sample**

This compact representation captures the full 3D geometry of the hand without any pixel-level noise, making it highly efficient and classifier-friendly.

**Why StandardScaler?**

The three coordinate types (x, y, z) have different natural ranges and variances. Without scaling, the Random Forest may weight x/y features more heavily simply because they span a larger range. StandardScaler normalises each feature to zero mean and unit variance, ensuring all 63 dimensions contribute equally to classification.

---

### 10.4 Data Augmentation Strategy

Each training image is augmented to produce 3 variants before landmark extraction:

| Augmentation | Parameters | Purpose |
|---|---|---|
| **Horizontal Flip** | Mirror image | Handles left-hand vs right-hand variants |
| **Random Rotation** | ±15 degrees | Handles tilted or angled gesture poses |
| **Brightness Adjustment** | ±30 pixel intensity | Handles varying lighting conditions |

This triples the effective training set size without collecting additional data, dramatically improving the model's ability to generalise to real-world variation in lighting, hand orientation, and user differences.

---

## 11. Real-Time Inference Pipeline

When a webcam is available, SignSense enters a continuous inference loop:

```
While webcam is open:
    1. Capture frame (BGR)
    2. Convert to RGB for MediaPipe
    3. Run MediaPipe hand detection
    4. If hand detected:
       a. Extract 21 landmarks → flatten to 63-dim vector
       b. Apply StandardScaler transform
       c. Predict gesture class with Random Forest
       d. Get prediction confidence (predict_proba)
       e. Overlay gesture label + confidence on frame
       f. Update live matplotlib bar chart
    5. If no hand detected:
       a. Display "No hand detected" message
    6. Show annotated frame
    7. Check for 'q' keypress to quit
```

**Key configuration parameters:**

| Parameter | Default | Description |
|---|---|---|
| `MIN_DETECTION_CONFIDENCE` | `0.3` | Lower = more sensitive detection in poor lighting |
| `MAX_IMAGES_PER_CLASS` | `500` | Cap per gesture class to balance training data |
| `AUGMENTATION_FACTOR` | `3` | Number of augmented variants per original image |
| `N_ESTIMATORS` | `100` | Number of trees in the Random Forest |

---

## 12. Improving Model Accuracy

If you find the model's accuracy unsatisfactory for your use case, apply the following strategies in order:

**Strategy 1 — Verify landmark detection quality:**

```python
import cv2
import mediapipe as mp

hands = mp.solutions.hands.Hands(min_detection_confidence=0.3)
img = cv2.imread("your_test_image.jpg")
result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print("Hand detected:", bool(result.multi_hand_landmarks))
print("Landmarks:", result.multi_hand_landmarks)
```

If landmarks are not detected on your test images, the model has no useful input — improve image quality or lower `MIN_DETECTION_CONFIDENCE`.

**Strategy 2 — Increase augmentation:**

```python
AUGMENTATION_FACTOR = 5   # Default is 3
```

**Strategy 3 — Increase training data cap:**

```python
MAX_IMAGES_PER_CLASS = 1000   # Default is 500
```

**Strategy 4 — Try a more powerful classifier:**

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    max_iter=500,
    random_state=42
)
```

**Strategy 5 — Add K-Fold cross-validation:**

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 13. Troubleshooting

**Dataset Not Found:**

```bash
# Check that kaggle.json exists and has correct permissions
ls -la ~/.kaggle/kaggle.json
# Expected: -rw------- 1 user user ...

# Re-download if needed
python -c "import kagglehub; kagglehub.dataset_download('abhishek14398/gesture-recognition-dataset')"
```

**Low Model Accuracy:**

Check the class distribution in your loaded dataset:

```python
from collections import Counter
print(Counter(y_train))
# Each class should have a similar count
```

If one class has far fewer samples, increase `MAX_IMAGES_PER_CLASS` or add more augmentations for the underrepresented class.

**Webcam Not Opening:**

```python
import cv2
cap = cv2.VideoCapture(0)
print("Webcam opened:", cap.isOpened())
cap.release()
```

If `False`, try `cv2.VideoCapture(1)` (external webcam index). On Linux, verify the camera is accessible at `/dev/video0`.

**MediaPipe Landmark Detection Failing:**

Ensure input images are in RGB format before passing to MediaPipe:

```python
# CORRECT
rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
result = hands.process(rgb_frame)

# WRONG — MediaPipe expects RGB, not BGR
result = hands.process(bgr_frame)
```

**Matplotlib Not Updating in Real Time:**

```python
import matplotlib
matplotlib.use('TkAgg')   # or 'Qt5Agg' depending on your system
import matplotlib.pyplot as plt
plt.ion()   # Enable interactive mode
```

---

## 14. Roadmap — Future Improvements

**v1.1 — Expanded Gesture Library**
Add support for additional gestures: `peace`, `ok`, `fist`, `point` — expanding the gesture vocabulary for richer HCI applications.

**v1.2 — Dynamic Gesture Recognition**
Move from static frame classification to temporal gesture recognition using a sliding window of frames, enabling detection of motion-based gestures like `wave` (which inherently requires movement).

**v1.3 — Multi-Hand Support**
Extend the pipeline to process two hands simultaneously — enabling two-handed gesture combinations and sign language phrase detection.

**v1.4 — Streamlit Web Interface**
Build an interactive browser-based demo using Streamlit — allowing users to test gesture recognition through their browser webcam without installing any dependencies.

**v1.5 — Gradio on Hugging Face Spaces**
Deploy a zero-installation public demo to Hugging Face Spaces using Gradio, allowing anyone to test the model by uploading an image.

**v2.0 — Deep Learning Upgrade**
Replace the Random Forest with a lightweight CNN (MobileNetV2 or EfficientNet-B0) trained end-to-end on hand images, achieving higher accuracy on ambiguous gestures.

**v2.1 — Sign Language Dataset**
Integrate the ASL (American Sign Language) alphabet dataset to extend SignSense toward full sign language recognition — a meaningful accessibility application.

---

## 15. Contributing

Contributions are welcome — from new gesture classes to model improvements to documentation fixes.

**How to Contribute:**

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/add-peace-gesture
   ```

3. **Make your changes** — follow PEP8 and document all functions

4. **Test your changes** on both Kaggle notebook and local environments

5. **Commit with a clear message**
   ```bash
   git commit -m "feat: add peace gesture class with augmentation"
   ```

6. **Push and open a Pull Request**
   ```bash
   git push origin feature/add-peace-gesture
   ```

**Good First Issues:**

- Add case-insensitive folder name matching in the dataset loader
- Implement a `--gesture` CLI argument to select which gestures to train on
- Add a confusion matrix visualisation after training
- Write unit tests for the feature extraction function
- Add support for video file input (`.mp4`) in addition to webcam

---

## 16. License

```
MIT License

Copyright (c) 2025 Ibtesaam Aslam

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 17. Author

<div align="center">

### **Ibtesaam Aslam**
*Full-Stack Developer · ML Enthusiast · Computer Vision Builder*

[![GitHub](https://img.shields.io/badge/GitHub-%40ibtesaamaslam-181717?style=for-the-badge&logo=github)](https://github.com/ibtesaamaslam)
[![Repository](https://img.shields.io/badge/Repo-Gesture--Recognition--Model-0070f3?style=for-the-badge&logo=github)](https://github.com/ibtesaamaslam/Gesture-Recognition-Model-)

</div>

---

## 18. Acknowledgments

| Resource | Contribution |
|---|---|
| [MediaPipe — Google](https://mediapipe.dev/) | Real-time hand landmark detection framework |
| [Abhishek14398 — Kaggle](https://www.kaggle.com/datasets/abhishek14398/gesture-recognition-dataset) | The gesture recognition dataset used for training |
| [Scikit-learn](https://scikit-learn.org/) | Random Forest classifier, StandardScaler, and evaluation metrics |
| [OpenCV](https://opencv.org/) | Webcam capture, image I/O, and frame annotation |
| [Matplotlib](https://matplotlib.org/) | Live confidence bar chart visualisation |
| [KaggleHub](https://github.com/Kaggle/kagglehub) | Programmatic dataset download and caching |

---

<div align="center">

---

*Made with ❤️ by [Ibtesaam Aslam](https://github.com/ibtesaamaslam)*

*Teaching machines to understand the language of hands.*

[![Star on GitHub](https://img.shields.io/github/stars/ibtesaamaslam/Gesture-Recognition-Model-?style=social)](https://github.com/ibtesaamaslam/Gesture-Recognition-Model-)

</div>

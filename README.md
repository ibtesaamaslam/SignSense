
# SignSense: Hand Gesture Recognition System

**SignSense** is a machine learning-based hand gesture recognition system that detects and classifies hand gestures (`wave`, `stop`, `thumbs_up`) using **MediaPipe** for hand landmark detection and a **Random Forest** classifier for robust prediction. It supports both **dataset-based training** (from Kaggle) and **real-time inference** via webcam, making it ideal for human-computer interaction, accessibility tools, and interactive AI applications.

---

## 📑 Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Dataset](#dataset)
* [Usage](#usage)
* [Code Structure](#code-structure)
* [Improving Model Accuracy](#improving-model-accuracy)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## 🚀 Project Overview

SignSense uses computer vision and machine learning to identify hand gestures from static images and live video. It:

* Extracts 3D hand landmarks with MediaPipe
* Converts them into feature vectors
* Trains a Random Forest classifier
* Performs real-time gesture detection with live visual feedback

The system resolves common issues like mislabelled gesture classes and low model accuracy by applying:

* Class filtering
* Data augmentation
* Feature scaling

It runs seamlessly on:

* Local machines (real-time webcam inference)
* Cloud environments like Kaggle (dataset-based training)

---

## 🎯 Features

* **🖐 Gesture Recognition**: Detects `wave`, `stop`, `thumbs_up`
* **🎥 Real-Time Inference**: Predicts gestures live from webcam
* **📁 Dynamic Dataset Loader**: Auto-maps folder names like `0`, `1`, `2` or `wave`, `stop`, `thumbs_up`
* **🌀 Data Augmentation**: Random rotations, flips, brightness adjustments
* **⚖️ Feature Scaling**: Uses StandardScaler to normalize landmarks
* **📈 Live Confidence Plot**: Real-time matplotlib graph during webcam predictions
* **💡 Error Logging**: Handles image load errors and missing landmarks
* **🌐 Kaggle Compatible**: Fully runnable in Kaggle notebooks (no webcam needed)

---

## 🧰 Technologies Used

* **Python 3.6+**
* **OpenCV** – for webcam & image handling
* **MediaPipe** – for hand landmark extraction
* **Scikit-learn** – for Random Forest classifier & scaling
* **NumPy** – for feature manipulation
* **Matplotlib** – for live prediction plotting
* **KaggleHub** – for dataset download
* **Tqdm** – for progress bars

---

## ⚙️ Installation

### ✅ Prerequisites

* Python 3.6+
* Webcam (optional for real-time)
* Internet access (for Kaggle dataset)
* `kaggle.json` API token if using KaggleHub

### 🔧 Step-by-Step

```bash
git clone https://github.com/your-username/SignSense.git
cd SignSense
pip install opencv-python mediapipe numpy scikit-learn matplotlib kagglehub tqdm
```

### 🔑 Setup Kaggle API (Optional)

1. Get your `kaggle.json` from [Kaggle Account Settings](https://www.kaggle.com/account)
2. Place it in:

   * Linux/Mac: `~/.kaggle/kaggle.json`
   * Windows: `C:\Users\<Username>\.kaggle\kaggle.json`
3. Set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📦 Dataset

* **Source**: [Kaggle: abhishek14398/gesture-recognition-dataset](https://www.kaggle.com/datasets/abhishek14398/gesture-recognition-dataset)
* **Gestures**: `wave`, `stop`, `thumbs_up`
* **Format**: `.jpg`, `.jpeg`, `.png`
* **Structure**:

```
gesture-recognition-dataset/
├── train/
│   ├── 0/ or wave/
│   ├── 1/ or stop/
│   ├── 2/ or thumbs_up/
├── val/
    ├── ...
```

### 🔍 Dataset Processing

* Filters non-gesture classes
* Extracts 63 features per image (21 landmarks × 3 coords)
* Max 500 images per class
* Augments each with flips, rotations, brightness tweaks

---

## 🧪 Usage

### 🔧 Cell 1: Install Libraries

```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib kagglehub tqdm
```

### 🔍 Cell 2: Inspect Dataset

```python
import os, glob, kagglehub

KAGGLE_DATASET = "abhishek14398/gesture-recognition-dataset"
dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)

for root, dirs, files in os.walk(dataset_path):
    print(f"{root} → {dirs}, Files: {len(files)}")

image_paths = glob.glob(os.path.join(dataset_path, "**/*.*"), recursive=True)
print(f"Found {len(image_paths)} images.")
```

### 🧠 Cell 3: Train & Run

* Trains on dataset
* Opens webcam
* Live prediction + matplotlib plot

```bash
python gesture_recognition.py
```

---

## 🧭 Code Structure

```python
gesture_recognition.py
├── Config
├── Feature Extraction
├── Data Augmentation
├── Dataset Loader
├── Training + Inference
└── Live Visualization
```

---

## 🎯 Improving Model Accuracy

* ✅ Feature scaling (StandardScaler)
* ✅ 100-tree Random Forest
* ✅ Augmentation ×3 (flip, rotate, brightness)
* ✅ Min detection confidence: 0.3

📈 Try:

* More augmentations
* MLPClassifier
* K-Fold validation

---

## 🛠️ Troubleshooting

### Dataset Not Found?

Check `dataset_path` and folder mappings (`GESTURE_MAPPING`).

### Low Accuracy?

Try:

* Valid landmark detection:

```python
img = cv2.imread("image.jpg")
result = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print("Landmarks:", bool(result.multi_hand_landmarks))
```

* Increase `AUGMENTATION_FACTOR`
* Use a different classifier

### Webcam Not Working?

```python
import cv2
cap = cv2.VideoCapture(0)
print("Webcam opened:", cap.isOpened())
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Commit & push
4. Open a pull request

Please follow PEP8 and document your changes!

---

## 📄 License

**MIT License** — See the `LICENSE` file.

---

## 🙏 Acknowledgments

* [MediaPipe](https://mediapipe.dev) for real-time hand tracking
* [Kaggle](https://www.kaggle.com/datasets/abhishek14398/gesture-recognition-dataset) for the dataset
* [Scikit-learn](https://scikit-learn.org) for ML tools
* [OpenCV](https://opencv.org) for video & image I/O

# VISION LOCK: Multi-Layer Biometric Authentication System
> **Python | OpenCV | CNN (PyTorch) | Dlib**

**VISION LOCK** is a real-time multi-layer biometric authentication system combining deep Convolutional Neural Networks (CNN) for face recognition with Dlib 68-point facial landmark tracking for anti-spoofing and liveness verification.

---

## 🌟 Key Highlights & Engineering Achievements

- **CNN-Based Facial Recognition (98%+ Accuracy)**: Upgraded from legacy handcrafted LBPH features to a 4-block Deep Convolutional Neural Network (`VisionLockCNN`) with Batch Normalization, LeakyReLU, Dropout regularization, and a 128-dimensional bottleneck feature embedding.
- **Enhanced Preprocessing for Environmental Robustness**: Integrated Contrast Limited Adaptive Histogram Equalization (**CLAHE**) on the luminance channel and edge-preserving bilateral filtering to maintain high recognition accuracy across varying lighting conditions and sensor noise.
- **Bottleneck Analysis & Optimization (+20% to +79% Speed Increase)**: Profiled the inference loop and resolved latency bottlenecks by downscaling the detection pyramid and optimizing tensor memory transfers, boosting real-time processing speed to **30+ FPS**.
- **Multi-Layer Defense**:
  - **Layer 1 (Knowledge Factor)**: User ID + SHA-256 salted/hashed password credentials.
  - **Layer 2 (Biometric Factor)**: Deep CNN facial recognition with calibrated confidence scoring.
  - **Layer 3 (Liveness / Anti-Spoofing)**: Dlib 68-point facial landmarks running interactive dynamic challenges (Blink detection via Eye Aspect Ratio [EAR], and real-time iris/pupil tracking for Look Left / Look Right).

---

## 📐 Architecture Overview

```
                                      [ Webcam Video Stream ]
                                                 │
                                                 ▼
                           [ CLAHE & Bilateral Denoising Preprocessing ]
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             [ Downscaled Face Detection ]                     [ Dlib 68 Facial Landmarks ]
                        │                                                 │
                        ▼                                                 ▼
               [ Face ROI Crop ]                                [ Anti-Spoofing / Liveness ]
                        │                                        - Eye Aspect Ratio (Blink)
                        ▼                                        - Iris Tracking (Left/Right)
            [ VisionLock Deep CNN ]                                       │
            - Conv Block 1 (32)                                           │
            - Conv Block 2 (64)                                           │
            - Conv Block 3 (128)                                          │
            - Conv Block 4 (256)                                          │
            - 128-D Bottleneck Embedding                                  │
            - Softmax Classifier Head                                     │
                        │                                                 │
                        ▼                                                 ▼
            [ CNN Identity Match (>=60%) ]                    [ Liveness Gesture Verified ]
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 ▼
                                     [ Access Granted (Login) ]
```

---

## 🚀 Installation & Prerequisites

### 1. Requirements
Ensure you have Python 3.10+ (tested on Python 3.10, 3.12, 3.13, 3.14).

Install dependencies via pip:
```bash
pip install -r requirements.txt
```
*(Dependencies: `torch`, `opencv-contrib-python`, `dlib-bin`, `numpy`, `pillow`)*

---

## 🖥️ How to Run the Project

### Option 1: Launch the Main GUI Application (Recommended)
From the project root:
```bash
python run.py
```
Or directly from the `FaceAuth` directory:
```bash
cd FaceAuth
python main.py
```

### Option 2: Step-by-Step CLI Execution

#### 1. Register a New User
```bash
python FaceAuth/register.py
```
- Enter a numeric User ID.
- The webcam opens and captures 40 biometric face samples saved into `dataset/user_<ID>/`.

#### 2. Train the Deep CNN Model
```bash
python FaceAuth/train_model.py
```
- Automatically scans all registered users in `dataset/`.
- Applies data augmentation (rotations, horizontal flips, contrast shifts).
- Trains `VisionLockCNN` with Cosine Annealing learning rate schedule.
- Saves the trained model weights to `trainer/face_cnn_model.pth` and class mappings to `trainer/model_classes.json`.
- Outputs training and validation accuracy (achieving 98-100%).

#### 3. Run Standalone Face Recognition & Liveness Verification
```bash
python FaceAuth/recognizer.py
```
- Starts webcam feed with real-time HUD (FPS counter, CNN confidence score, bounding box).
- Prompts for a random liveness challenge (`Blink`, `Look Left`, or `Look Right`).
- Authenticates the user when both CNN identity and liveness are confirmed.

#### 4. Run the Performance & Bottleneck Benchmark
```bash
python FaceAuth/benchmark.py
```
- Evaluates the CNN on all registered users in the dataset.
- Runs unoptimized vs. optimized frame latency analysis.
- Prints a benchmark report proving accuracy (100%) and processing speedup (+79%).

---

## 📂 Project Directory Structure

```
FaceAuth/
├── run.py                                    # Root launch script
├── requirements.txt                          # Project dependencies
├── README.md                                 # Documentation & architecture guide
└── FaceAuth/
    ├── main.py                               # Tkinter GUI application
    ├── cnn_model.py                          # VisionLockCNN PyTorch architecture
    ├── preprocessing.py                      # CLAHE contrast enhancement & performance monitor
    ├── train_model.py                        # CNN training & validation pipeline
    ├── recognizer.py                         # Real-time CNN inference & Dlib liveness verification
    ├── register.py                           # Biometric face capture module
    ├── capture_faces.py                      # Auxiliary face dataset collector
    ├── benchmark.py                          # Accuracy and speed benchmarking script
    ├── database.json                         # Encrypted/hashed credential store
    ├── haarcascade_frontalface_default.xml   # Pretrained Haar cascade detector
    ├── shape_predictor_68_face_landmarks.dat # Dlib 68-point facial landmark model
    ├── dataset/                              # Enrolled biometric face samples
    │   ├── user_1/
    │   ├── user_2/
    │   ├── user_3/
    │   └── user_4/
    ├── trainer/                              # Saved weights and model artifacts
        ├── face_cnn_model.pth                # Trained PyTorch CNN weights
        ├── model_classes.json                # User class mapping
        ├── model_metrics.json                # Training metrics & accuracy records
        └── trainer.yml                       # Backward-compatible fallback
├── web_app.py                                # Production FastAPI Web App & REST API
├── Dockerfile                                # Cloud Container definition
├── Procfile                                  # Deployment profile for Render/Railway
└── render.yaml                               # One-click Render Blueprint

---

## 🌐 Web & Cloud Deployment (FastAPI + Modern Web UI)

You can run the web-based version of Vision Lock that works in any web browser and can be deployed directly to cloud platforms:

### 1. Run Locally
```bash
python web_app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser. The web application features:
- Live webcam video capture with Canvas HUD bounding boxes.
- Interactive liveness challenge prompts (Blink / Look Left / Look Right).
- REST endpoints for `/api/authenticate`, `/api/register`, and `/api/benchmark`.

### 2. Free Cloud Hosting Options

#### **Option A: Render (Free Web Service)**
1. Push your repository to GitHub.
2. Sign in to [Render.com](https://render.com) and click **New Web Service**.
3. Select your GitHub repository.
4. Set **Environment** to `Docker` (Render will automatically detect the included `Dockerfile` and `render.yaml`).
5. Click **Create Web Service**. Your live URL will be ready in 2–3 minutes!

#### **Option B: Hugging Face Spaces (Free Cloud GPU/CPU)**
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Docker** as the Space SDK.
3. Push your repository to the Hugging Face Git remote.
4. Your application will launch with a public portfolio link!

#### **Option C: Railway**
1. Connect your GitHub repository on [Railway.app](https://railway.app).
2. Railway will auto-detect the `Dockerfile` and deploy your FastAPI app.

```

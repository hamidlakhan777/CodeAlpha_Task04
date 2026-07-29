# CodeAlpha_Task04
# 👁️ NexaTrace AI — Real-Time Object Detection & Tracking System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo)
![OpenCV](https://img.shields.io/badge/OpenCV-Headless-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**NexaTrace AI** is a lightweight, futuristic Computer Vision dashboard designed for high-speed object detection, persistent multi-object tracking (MOT), and live trajectory visualization. Built with an optimized **YOLOv8** engine and a Cyberpunk-inspired **Streamlit** UI, it provides real-time video analytics while remaining lightweight enough for serverless cloud deployment.

---

## ✨ Key Features

- **⚡ Lightweight Neural Engine:** Powered by `YOLOv8n` for ultra-fast frame inference with low memory overhead.
- **🏷️ Persistent Tracking IDs:** Assigns persistent track IDs (`ID: #X`) across consecutive video frames using built-in multi-object tracking.
- **📍 Trajectory Line Mapping:** Real-time path tracing to visually track object movement history.
- **🎛️ Dynamic Cyberpunk Dashboard:**
  - Real-time confidence threshold adjustment.
  - Interactive multi-class selection (Pedestrians, Vehicles, Animals, etc.).
  - Live KPI metrics for active detections, total track counts, and system FPS.
- **☁️ Cloud & Vercel Optimized:** Configured with `opencv-python-headless` to eliminate GUI dependencies for smooth serverless builds.

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Frontend / Dashboard:** Streamlit (Custom Glassmorphism & Neon CSS)
- **Object Detection & Tracking:** Ultralytics YOLOv8 (`yolov8n.pt`)
- **Computer Vision:** OpenCV Headless (`opencv-python-headless`)
- **Image Processing:** PIL & NumPy

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed (v3.9 or higher recommended).

### 2. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/NexaTrace-AI.git](https://github.com/YOUR_USERNAME/NexaTrace-AI.git)
cd NexaTrace-AI
Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Application
Bash
streamlit run app.py
📂 Project Structure
Plaintext
📁 NexaTrace-AI/
│── 📄 app.py              # Main Streamlit application entry point
│── 📄 requirements.txt    # Production dependencies
│── 📄 README.md           # Project documentation
└── 📄 yolov8n.pt          # Auto-downloaded YOLOv8 Nano model weights

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
from collections import defaultdict
import tempfile
import os

# Set Streamlit Page Config
st.set_page_config(page_title="Neural Tracker", page_icon="👁️", layout="wide")

# CSS Styling (Futuristic Cyberpunk)
css = """
<style>
/* Base theme */
.stApp {
    background-color: #0B0E14;
    color: #E2E8F0;
    font-family: 'Courier New', Courier, monospace;
}
[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 2px solid #8A2BE2;
}
/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: #00E699 !important;
    text-shadow: 0 0 10px rgba(0, 230, 153, 0.5);
    letter-spacing: 2px;
}
/* KPIs */
.metric-container {
    display: flex;
    justify-content: space-around;
    padding: 10px 0 30px 0;
}
.metric-card {
    background: rgba(22, 27, 34, 0.8);
    border: 1px solid #00E699;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    width: 30%;
    box-shadow: 0 0 15px rgba(0, 230, 153, 0.2);
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}
.metric-card:hover {
    box-shadow: 0 0 25px rgba(138, 43, 226, 0.6);
    transform: translateY(-5px);
    border-color: #8A2BE2;
}
.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: #00E699;
    text-shadow: 0 0 10px #00E699;
}
.metric-label {
    font-size: 1rem;
    color: #8A2BE2;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 5px;
}
/* Glowing Pulse Animation */
@keyframes pulse {
    0% { text-shadow: 0 0 5px #00E699; }
    50% { text-shadow: 0 0 20px #00E699, 0 0 30px #00E699; }
    100% { text-shadow: 0 0 5px #00E699; }
}
.pulse-text {
    animation: pulse 2s infinite;
}
/* Scanning bar on video container */
[data-testid="stImage"] {
    position: relative;
    overflow: hidden;
    border: 2px solid #8A2BE2;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(138, 43, 226, 0.3);
}
[data-testid="stImage"]::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: rgba(0, 230, 153, 0.8);
    box-shadow: 0 0 15px #00E699;
    animation: scan 3s linear infinite;
    pointer-events: none;
}
@keyframes scan {
    0% { top: 0; }
    50% { top: 100%; }
    100% { top: 0; }
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Using yolov8n.pt - lightweight and fast
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar Setup
st.sidebar.markdown("<h2 style='text-align: center;'>CONTROL PANEL</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

# Map model classes
coco_classes = model.names
class_options = list(coco_classes.values())
default_classes = [c for c in ["person", "car", "bicycle", "bus", "dog"] if c in class_options]

selected_classes = st.sidebar.multiselect("Object Classes", class_options, default=default_classes)

# Get the IDs for the selected classes to pass into YOLO
selected_class_ids = [k for k, v in coco_classes.items() if v in selected_classes]

show_trajectory = st.sidebar.checkbox("Enable Trajectory Lines", value=True)

st.sidebar.markdown("---")
st.sidebar.info("Upload an image or video to begin neural tracking. The dashboard will update in real-time.")

# Main Interface
st.markdown("<h1 class='pulse-text' style='text-align: center;'>NEURAL VISION // TRACKER</h1>", unsafe_allow_html=True)

# KPI Dashboard Placeholder
kpi_container = st.empty()

def render_kpis(objects=0, tracks=0, fps=0.0):
    html = f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-value">{objects}</div>
            <div class="metric-label">Objects Detected</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{tracks}</div>
            <div class="metric-label">Active Tracks</div>
        </div>
        <div class="metric-card">
            <div class="metric-value pulse-text">{fps:.1f}</div>
            <div class="metric-label">System FPS</div>
        </div>
    </div>
    """
    kpi_container.markdown(html, unsafe_allow_html=True)

render_kpis()

upload_type = st.radio("Select Input Mode", ("Image", "Video"), horizontal=True)
uploaded_file = st.file_uploader(f"Upload {upload_type}", type=["jpg", "png", "jpeg", "mp4", "avi", "mov"])

# Tracking history for trajectories
track_history = defaultdict(lambda: [])

def process_frame(frame, results):
    current_ids = set()
    total_objects = 0
    
    if results and results[0].boxes:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        track_ids = None
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy()

        for i, box in enumerate(boxes):
            cls = int(clss[i])
            
            if selected_class_ids and cls not in selected_class_ids:
                continue
                
            conf = confs[i]
            if conf < conf_threshold:
                continue

            total_objects += 1
            x1, y1, x2, y2 = map(int, box)
            
            # Colors (BGR for OpenCV)
            neon_cyan = (153, 230, 0)
            neon_purple = (226, 43, 138)
            color = neon_cyan if cls == 0 else neon_purple 
            
            # === CHANGED:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            # Construct label
            label = f"{coco_classes[cls]} {conf:.2f}"
            if track_ids is not None:
                track_id = int(track_ids[i])
                label += f" ID:#{track_id}"
                current_ids.add(track_id)
                
                # Trajectory calculation
                if show_trajectory:
                    center = (int((x1 + x2)/2), int((y1 + y2)/2))
                    track_history[track_id].append(center)
                    if len(track_history[track_id]) > 40:
                        track_history[track_id].pop(0)
                        
                    points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                    # Trajectory line thin (1px)
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=1)
            
            # === CHANGED:
            font_scale = 0.35
            font_thickness = 1
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            # Label background height & text offset reduced
            cv2.rectangle(frame, (x1, y1 - h - 6), (x1 + w + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            
    return frame, total_objects, len(current_ids)
    
if uploaded_file is not None:
    if upload_type == "Image":
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        start_time = time.time()
        
        # Inference (Image doesn't need tracking usually, but we can call it)
        classes_arg = selected_class_ids if selected_class_ids else None
        results = model(frame, conf=conf_threshold, classes=classes_arg)
        
        proc_frame, objs, tracks = process_frame(frame, results)
        
        fps = 1.0 / (time.time() - start_time + 1e-9)
        render_kpis(objs, tracks, fps)
        
        proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        st.image(proc_frame, use_container_width=True)
        
    elif upload_type == "Video":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            classes_arg = selected_class_ids if selected_class_ids else None
            # Use model.track() for video to assign persistent IDs
            results = model.track(frame, persist=True, conf=conf_threshold, classes=classes_arg)
            
            proc_frame, objs, tracks = process_frame(frame, results)
            
            fps = 1.0 / (time.time() - start_time + 1e-9)
            render_kpis(objs, tracks, fps)
            
            proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            stframe.image(proc_frame, channels="RGB", use_container_width=True)
            
        cap.release()
        os.remove(tfile.name)

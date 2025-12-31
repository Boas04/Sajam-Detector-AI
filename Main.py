import streamlit as st
from ultralytics import YOLO
import cv2
import math
import numpy as np
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Senjata Tajam AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS UNTUK TAMPILAN PROFESIONAL ---
st.markdown("""
    <style>
    /* Main Title */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #1F618D; /* Dark Blue */
        text-align: center;
        margin-bottom: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle */
    .sub-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 1.2rem;
        color: #566573;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* Project Box */
    .project-box {
        background: linear-gradient(135deg, #f6f8f9 0%, #e5ebee 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #1F618D;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 30px;
        font-family: 'Segoe UI', sans-serif;
    }
    .project-label {
        font-weight: bold;
        color: #1F618D;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .project-text {
        color: #2C3E50;
        font-size: 1.1rem;
        line-height: 1.5;
        font-weight: 600;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: bold;
        color: #7f8c8d;
    }
    div[data-testid="stMetricValue"] {
        color: #1F618D;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<p class="main-title">Mendeteksi Senjata Tajam</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Real-time Object Detection System with YOLO11</p>', unsafe_allow_html=True)

st.markdown("""
    <div class="project-box">
        <div class="project-label">Judul Proyek</div>
        <div class="project-text">Analisis Performansi YOLO11 dalam Mendeteksi Objek Berbahaya Ragam Senjata Tajam (Celurit, Parang, dan Golok) untuk Keamanan Lingkungan Area Publik</div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
st.sidebar.title("Panel Kontrol")
st.sidebar.markdown("---")

conf_threshold = st.sidebar.slider("🎚️ Tingkat Keyakinan (Confidence)", 0.0, 1.0, 0.50, 0.05)
st.sidebar.markdown("*Atur sensitivitas deteksi model.*")

st.sidebar.markdown("---")
st.sidebar.info("Model: **YOLO11 (best.pt)**\n\nClasses: Celurit, Golok, Parang")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

classNames = ["Celurit", "Golok", "Parang"]

# --- HELPER FUNCTIONS ---
def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    """Menggambar kotak dengan sudut yang lebih estetis"""
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

# --- MAIN APP LOGIC ---
col1, col2 = st.columns([3, 1])
with col1:
    run = st.checkbox('🔴 Aktifkan Kamera', value=False)
with col2:
    st.write("") # Spacer

# Placeholder untuk video dan stats
frame_placeholder = st.empty()
stats_container = st.container()

if run:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while run and cap.isOpened():
        success, img = cap.read()
        if not success:
            st.warning("Gagal membaca feed kamera.")
            break

        # Inference
        results = model(img, stream=True, conf=conf_threshold, verbose=False)
        
        detected_objects = []
        max_conf = 0.0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Confidence
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                
                # Class Name
                cls = int(box.cls[0])
                if cls < len(classNames):
                    current_class = classNames[cls]
                else:
                    current_class = "Unknown"
                
                detected_objects.append(current_class)

                # Warna dinamis berdasarkan kelas
                if current_class == "Celurit":
                    color = (0, 0, 255) # Merah
                elif current_class == "Golok":
                    color = (0, 140, 255) # Orange
                elif current_class == "Parang":
                    color = (255, 0, 255) # Magenta
                else:
                    color = (0, 255, 0) # Hijau

                # Gambar Kotak Fancy
                # cv2.rectangle(img, (x1, y1), (x2, y2), color, 1) # Kotak tipis dasar
                draw_fancy_box(img, (x1, y1), (x2, y2), color, 3, 15, 30)

                # Label Background yang rapi
                label = f'{current_class} {conf:.0%}'
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - 30), (x1 + w + 20, y1), color, -1)
                cv2.putText(img, label, (x1 + 10, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Overlay Status "Detected" ala Screenshot
        if detected_objects:
            # Kotak status di pojok kiri atas video
            overlay = img.copy()
            cv2.rectangle(overlay, (20, 20), (300, 85), (34, 139, 34), -1) # Forest Green background
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
            
            # Icon Checkmark (simulasi text)
            cv2.putText(img, "✔ Senjata Terdeteksi", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Detail objects
            obj_counts = {i:detected_objects.count(i) for i in set(detected_objects)}
            obj_text = ", ".join([f"{v} {k}" for k,v in obj_counts.items()])
            cv2.putText(img, obj_text, (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            
            # Border merah di sekeliling layar jika bahaya
            cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0, 0, 255), 5)

        # Tampilkan Video
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, channels="RGB", use_container_width=True)

        # Update Metrics Real-time
        with stats_container:
            m1, m2, m3 = st.columns(3)
            m1.metric("Max Confidence", f"{max_conf:.1%}", delta_color="normal")
            m2.metric("Objek Terdeteksi", len(detected_objects), delta_color="inverse")
            m3.metric("Model AI", "YOLO11 - Nano")

    cap.release()
else:
    # Tampilan saat kamera mati
    frame_placeholder.markdown("""
        <div style="background-color: #f0f2f6; padding: 50px; border-radius: 10px; text-align: center; border: 2px dashed #ccc;">
            <h3 style="color: #555;">Kamera Non-Aktif</h3>
            <p>Klik checkbox <b>'Aktifkan Kamera'</b> di atas untuk memulai deteksi.</p>
        </div>
    """, unsafe_allow_html=True)

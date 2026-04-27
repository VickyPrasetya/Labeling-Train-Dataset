import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch

# Konfigurasi halaman
st.set_page_config(page_title="VISIOPED - AI Analysis", layout="wide")

# 1. Load Model Hasil Training (Path sesuai screenshot folder kamu)
@st.cache_resource
def load_model():
    # Menggunakan r'' agar path Windows dibaca dengan benar
    model_path = r'D:\Data Vicky\Dataset Foot Ulcer\runs\detect\VISIOPED\training_ulang\weights\best.pt'
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model best.pt. Pastikan file ada di: {e}")

st.title("🩺 VISIOPED: Sistem Skrining Gangguan Kesehatan")
st.write("Inovasi Analisis Visual Kaki Berbasis Algoritma YOLOv11")
st.markdown("---")

# 2. Fitur Upload Foto
uploaded_file = st.file_uploader("Unggah foto kaki untuk dianalisis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Pra-proses Gambar
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Layout Kolom
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Hasil Deteksi YOLOv11")
        # Jalankan Prediksi (Inferensi) dengan confidence 0.20 agar lebih sensitif
        results = model.predict(source=img_cv, conf=0.20) 
        
        # Plot Bounding Box
        res_plotted = results[0].plot()
        # Mengubah ukuran gambar agar tidak terlalu besar (width=400)
        st.image(res_plotted, channels="BGR", width=400)

    with col2:
        st.subheader("🔍 XAI Classification (Heatmap)")
        with st.spinner("Menganalisis area fitur..."):
            # Implementasi XAI Visual sederhana berdasarkan koordinat bounding box
            mask = np.zeros(img_cv.shape[:2], np.uint8)
            
            # Mendapatkan koordinat box untuk membuat heatmap
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                # Memberikan bobot visual pada area yang dideteksi
                mask[y1:y2, x1:x2] = 255
            
            # Membuat efek Heatmap halus
            if np.any(mask > 0):
                mask_blurred = cv2.GaussianBlur(mask, (71, 71), 0)
                heatmap = cv2.applyColorMap(mask_blurred, cv2.COLORMAP_JET)
                xai_output = cv2.addWeighted(img_cv, 0.7, heatmap, 0.3, 0)
                # Mengubah ukuran gambar agar sama dengan hasil deteksi (width=400)
                st.image(xai_output, channels="BGR", width=400)
                st.info("Area merah menunjukkan fokus utama AI dalam melakukan klasifikasi.")
            else:
                st.warning("Tidak ada area spesifik yang cukup kuat untuk dianalisis XAI.")

    # 3. Panel Informasi Hasil
    st.markdown("---")
    st.subheader("📊 Laporan Diagnosa")
    
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Mapping nama kelas agar lebih informatif
            st.success(f"**Temuan:** {class_name.upper()} (Tingkat Keyakinan: {conf_score:.2%})")
            
            # Contoh penjelasan XAI sederhana berbasis teks
            if class_name.lower() == 'diabetes' or class_id == 2: # Sesuaikan dengan ID kelasmu
                st.write("**Penjelasan AI:** Ditemukan pola anomali pada integritas kulit yang merujuk pada indikasi diabetes.")
    else:
        st.error("Sistem tidak mendeteksi adanya kelainan pada gambar yang diunggah.")

# Footer
st.markdown("<br><hr><center>VISIOPED © 2026 - TRPL Politeknik Astra</center>", unsafe_allow_stdio=True)
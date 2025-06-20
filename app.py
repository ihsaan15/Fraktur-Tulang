import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
import os

# =================================================================
# BAGIAN 1: MUAT MODEL DAN SCALER
# =================================================================
# Pastikan file 'model_fraktur_terbaik.pkl' dan 'scaler.pkl' berada di direktori yang sama
# dengan file app.py ini, atau berikan path lengkapnya.

try:
    model = joblib.load('./models/model_fraktur_terbaik.pkl')
    scaler = joblib.load('./models/scaler.pkl')
    st.sidebar.success("Model dan Scaler berhasil dimuat!")
except FileNotFoundError:
    st.sidebar.error("Error: File model atau scaler tidak ditemukan.")
    st.sidebar.info("Pastikan 'model_fraktur_terbaik.pkl' dan 'scaler.pkl' ada di direktori ini.")
    model = None
    scaler = None

# =================================================================
# BAGIAN 2: FUNGSI EKSTRAKSI FITUR (SESUAI DENGAN NOTEBOOK)
# =================================================================

def process_and_extract_features(image, target_size=(200, 200)):
    """
    Memproses citra dan mengekstrak fitur GLCM dan Hu Moments.
    """
    # Konversi ke grayscale jika gambar berwarna
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image # Gambar sudah grayscale

    # --- LANGKAH PENTING: IMAGE RESIZING ---
    img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)

    # Pipeline Pemrosesan Citra
    img_denoised = cv2.medianBlur(img_resized, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)

    # --- Ekstraksi Fitur Tekstur dari gambar CLAHE (GLCM) ---
    # Perhatikan penanganan jika gambar terlalu kecil
    if img_clahe.shape[0] < 5 or img_clahe.shape[1] < 5:
         st.warning("Gambar terlalu kecil untuk GLCM dengan distance=5. Menggunakan distance=1.")
         glcm = graycomatrix(img_clahe, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    else:
         glcm = graycomatrix(img_clahe, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)


    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # --- Ekstraksi Fitur Bentuk dari Tepi Canny ---
    img_edges = cv2.Canny(img_clahe, 50, 250)
    contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    jumlah_kontur = len(contours)

    # Hitung Hu Moments dari kontur terbesar
    hu_moments = np.zeros(7)
    if jumlah_kontur > 0:
        main_contour = max(contours, key=cv2.contourArea)
        # Periksa apakah kontur memiliki setidaknya 5 titik untuk menghindari error moments
        if len(main_contour) >= 5:
             moments = cv2.moments(main_contour)
             # Pastikan momen tidak nol sebelum menghitung Hu Moments
             if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments).flatten()
             else:
                 st.warning("Tidak dapat menghitung Hu Moments untuk kontur ini (m00 is zero).")
        else:
            st.warning(f"Kontur terbesar terlalu kecil ({len(main_contour)} titik) untuk menghitung Hu Moments.")


    # Urutan fitur HARUS sama dengan urutan saat pelatihan
    vektor_fitur = [
        contrast, dissimilarity, homogeneity, energy, correlation,
        hu_moments[0], hu_moments[1], hu_moments[2], jumlah_kontur
    ]

    return np.array(vektor_fitur).reshape(1, -1), img_resized, img_denoised, img_clahe, img_edges

# =================================================================
# BAGIAN 3: TAMPILAN STREAMLIT
# =================================================================

st.title("Deteksi Fraktur pada Tibia dan Fibula")
st.write("Upload citra X-ray tulang kaki (tibia/fibula) untuk mendeteksi fraktur.")

# Upload file gambar
uploaded_file = st.file_uploader("Pilih citra X-ray...", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # Baca gambar menggunakan OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_color is not None:
        st.subheader("Citra yang Diunggah")
        st.image(img_color, channels="BGR", caption="Citra Asli", use_container_width=True)

        if model is not None and scaler is not None:
            st.subheader("Hasil Deteksi Fraktur")

            # Proses gambar dan ekstrak fitur
            with st.spinner("Memproses citra dan mengekstrak fitur..."):
                 features, img_resized, img_denoised, img_clahe, img_edges = process_and_extract_features(img_color)

            if features is not None:
                # Penskalaan fitur menggunakan scaler yang sudah dimuat
                features_scaled = scaler.transform(features)

                # Prediksi menggunakan model
                prediction = model.predict(features_scaled)
                probability = model.predict_proba(features_scaled)

                # Tampilkan hasil prediksi
                hasil_label = "FRAKTUR" if prediction[0] == 1 else "NORMAL (Tidak Fraktur)"
                kepercayaan = probability[0][prediction[0]] * 100

                if hasil_label == "FRAKTUR":
                    st.error(f"Hasil Klasifikasi: **{hasil_label}**")
                    st.write(f"Tingkat Kepercayaan: {kepercayaan:.2f}%")
                else:
                    st.success(f"Hasil Klasifikasi: **{hasil_label}**")
                    st.write(f"Tingkat Kepercayaan: {kepercayaan:.2f}%")

                # Opsi untuk menampilkan tahapan pemrosesan (opsional)
                if st.checkbox("Tampilkan Tahapan Pemrosesan Citra"):
                    st.subheader("Tahapan Pemrosesan Citra")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_resized, caption="1. Resized Grayscale", use_container_width=True)
                    with col2:
                        st.image(img_denoised, caption="2. Denoised (Median Filter)", use_container_width=True)

                    col3, col4 = st.columns(2)
                    with col3:
                        st.image(img_clahe, caption="3. Kontras (CLAHE)", use_container_width=True)
                    with col4:
                        st.image(img_edges, caption="4. Garis Tepi (Canny)", use_container_width=True)


            else:
                 st.error("Gagal mengekstrak fitur dari citra.")

        else:
            st.warning("Model atau scaler tidak berhasil dimuat. Prediksi tidak dapat dilakukan.")

    else:
        st.error("Gagal memuat citra. Pastikan file yang diunggah adalah format gambar yang valid.")
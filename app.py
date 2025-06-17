# =================================================================
# IMPOR LIBRARY
# =================================================================
import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops
import os

# =================================================================
# FUNGSI EKSTRAKSI FITUR (Sama seperti di Colab)
# =================================================================
def process_and_extract_features_lanjutan(image):
    # Konversi ke grayscale jika gambar berwarna
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image # Jika sudah grayscale

    # Pipeline Pemrosesan Citra
    img_denoised = cv2.medianBlur(img_gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)

    # --- Ekstraksi Fitur Tekstur dari gambar CLAHE (GLCM) ---
    glcm = graycomatrix(img_clahe, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # --- Ekstraksi Fitur Bentuk dari Tepi Canny ---
    # Pastikan gambar input Canny adalah 8-bit grayscale
    img_clahe_8bit = cv2.convertScaleAbs(img_clahe)
    img_edges = cv2.Canny(img_clahe_8bit, 100, 200) # Gunakan 8-bit image

    contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    jumlah_kontur = len(contours)

    # Hitung Hu Moments dari kontur terbesar
    hu_moments = np.zeros(7)
    if jumlah_kontur > 0:
        main_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(main_contour)
        # Mencegah error jika momen bernilai 0
        if moments['mu02'] != 0 and moments['mu20'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
        else:
             hu_moments = np.zeros(7) # Set ke nol jika momen tidak valid


    return {
        'contrast': contrast, 'dissimilarity': dissimilarity, 'homogeneity': homogeneity,
        'energy': energy, 'correlation': correlation, 'hu_moment_1': hu_moments[0],
        'hu_moment_2': hu_moments[1], 'hu_moment_3': hu_moments[2], 'jumlah_kontur': jumlah_kontur
    }

# =================================================================
# FUNGSI MUAT MODEL & SCALER
# =================================================================
@st.cache_resource # Cache sumber daya agar model tidak dimuat berulang kali
def load_model_and_scaler():
    try:
        model_path = 'models/model_fraktur_terbaik.pkl' # Sesuaikan path
        scaler_path = 'models/scaler.pkl'              # Sesuaikan path
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: File model atau scaler tidak ditemukan. Pastikan 'model_fraktur_terbaik.pkl' dan 'scaler.pkl' ada di folder '{os.path.dirname(model_path)}'.")
        return None, None

# Muat model saat aplikasi dimulai
model, scaler = load_model_and_scaler()

# =================================================================
# BAGIAN STREAMLIT
# =================================================================
st.set_page_config(page_title="Deteksi Fraktur Tulang", layout="wide")

st.title("Deteksi Fraktur pada Tibia dan Fibula menggunakan Citra X-ray")
st.markdown("""
Aplikasi ini menggunakan Image Processing dan model Machine Learning untuk mendeteksi kemungkinan fraktur
berdasarkan citra X-ray tulang kaki bagian bawah (tibia dan fibula).
""")

# --- Unggah Gambar ---
st.header("Unggah Citra X-ray")
uploaded_file = st.file_uploader("Pilih file gambar...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED) # Baca dengan channel asli

    if image is None:
        st.error("Gagal membaca file gambar. Pastikan file yang diunggah adalah gambar yang valid.")
    else:
        st.image(image, caption="Citra Asli", use_container_width=True)

        st.header("Hasil Analisis")

        if model is not None and scaler is not None:
            # Proses dan ekstrak fitur
            st.write("Memproses citra dan mengekstraksi fitur...")
            features = process_and_extract_features_lanjutan(image)

            if features:
                st.write("Fitur berhasil diekstraksi.")
                # st.write(features) # Opsional: tampilkan fitur

                # Persiapan data untuk prediksi
                vektor_fitur = [
                     features['contrast'], features['dissimilarity'], features['homogeneity'],
                     features['energy'], features['correlation'], features['hu_moment_1'],
                     features['hu_moment_2'], features['hu_moment_3'], features['jumlah_kontur']
                ]
                fitur_array = np.array(vektor_fitur).reshape(1, -1)

                # Skalakan fitur menggunakan scaler yang dimuat
                fitur_scaled = scaler.transform(fitur_array)

                # Lakukan prediksi
                prediction = model.predict(fitur_scaled)
                probability = model.predict_proba(fitur_scaled)

                # Tampilkan hasil
                hasil_label = "FRAKTUR" if prediction[0] == 1 else "NORMAL (Tidak Fraktur)"
                kepercayaan = probability[0][prediction[0]] * 100

                st.subheader("Hasil Prediksi:")
                if hasil_label == "FRAKTUR":
                    st.error(f"Status: **{hasil_label}**")
                    st.error(f"Tingkat Kepercayaan: {kepercayaan:.2f}%")
                else:
                    st.success(f"Status: **{hasil_label}**")
                    st.success(f"Tingkat Kepercayaan: {kepercayaan:.2f}%")

                st.markdown("""
                <small>*Hasil ini adalah prediksi berdasarkan model machine learning dan
                tidak menggantikan diagnosis medis profesional.</small>
                """, unsafe_allow_html=True)

            else:
                st.error("Gagal mengekstraksi fitur dari citra.")
        else:
            st.warning("Model atau scaler belum dimuat. Pastikan file model (.pkl) ada.")

else:
    st.info("Silakan unggah gambar X-ray untuk memulai deteksi.")
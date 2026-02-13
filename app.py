"""
Deteksi Masker Wajah — Main Streamlit UI
=========================================
Clean modular UI. Semua logic ada di core/ dan utils/.
"""

import streamlit as st
from core.model_loader import load_model
from core.inference import run_inference, get_annotated_image
from utils.image_utils import prepare_image
from utils.result_utils import extract_results
from config import CONFIDENCE_THRESHOLD, IMAGE_EXTENSIONS

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Masker Wajah",
    page_icon="😷",
    layout="wide",
)

# ── Load custom CSS ──────────────────────────────────────────
with open("assets/styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    try:
        with open("sidebar.md") as md_file:
            st.markdown(md_file.read())
    except FileNotFoundError:
        st.info("File sidebar.md tidak ditemukan.")

# ── Model (cached — hanya load sekali) ──────────────────────
model = load_model()

# ── Header ───────────────────────────────────────────────────
st.title("😷 Deteksi Masker Wajah")
st.write("Upload gambar atau gunakan webcam untuk mendeteksi masker wajah.")

# ── Layout ───────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.5])

with col1:
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Gambar",
        type=IMAGE_EXTENSIONS,
    )

    # Webcam
    enable_webcam = st.checkbox("📷 Aktifkan Webcam")
    img_from_webcam = st.camera_input(
        "Ambil gambar dengan webcam",
        disabled=not enable_webcam,
    )

    st.divider()

    # Confidence slider
    threshold = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
    )

    # Reset button
    if st.button("Reset", use_container_width=True):
        st.rerun()

    # Legend
    st.markdown(
        """
        **Legend:**
        - 🟢 Pakai Masker
        - 🔴 Tanpa Masker
        - 🟡 Salah Pakai Masker
        """
    )


# ── Helper: proses & tampilkan hasil ─────────────────────────
def show_results(source_file, caption):
    """Proses gambar dan tampilkan hasil deteksi."""
    try:
        image_path = prepare_image(source_file)

        with st.spinner("⏳ Sedang menganalisis..."):
            results, inference_time = run_inference(model, image_path)

        # Gambar beranotasi
        annotated = get_annotated_image(results)
        st.image(annotated, caption=caption, use_container_width=True)

        # Tabel hasil
        df = extract_results(results)
        if not df.empty:
            df = df[df["Confidence"] >= threshold]
            st.subheader("Hasil Deteksi")
            st.table(df)
        else:
            st.info("Tidak ada objek yang terdeteksi.")

        st.caption(f"⏱️ Waktu proses: **{inference_time}** detik")

    except Exception as e:
        st.error(f"Error: {e}")


# ── Tampilkan hasil ──────────────────────────────────────────
with col2:
    if uploaded_file is not None:
        show_results(uploaded_file, "Hasil Deteksi — Upload")

    elif img_from_webcam is not None:
        show_results(img_from_webcam, "Hasil Deteksi — Webcam")
    else:
        st.info("Upload gambar atau aktifkan webcam untuk memulai deteksi.")
"""
Model loader dengan caching agar model hanya dimuat sekali.
"""

import streamlit as st
from ultralytics import YOLO
from config import MODEL_PATH


@st.cache_resource
def load_model():
    """
    Load model YOLO dari path yang dikonfigurasi.
    Model di-cache oleh Streamlit sehingga hanya dimuat sekali
    per session, bukan setiap kali halaman di-rerun.
    """
    return YOLO(MODEL_PATH)

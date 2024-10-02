import streamlit as st
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image
from image_extensions import IMAGE_EXTENSIONS

# Initialize YOLO model
model = YOLO('deteksi-masker-wajah.pt')

def process_image(image):
    # cek apakah file diupload
    if image is None:
        return 'No file uploaded', None
    
    try:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        # read file
        # image = Image.open(io.BytesIO(uploaded_file.read()))

        # hadeh perlu di convert ke rgb segala karena PIL error
        image = image.convert('RGB')

        save_filename = 'test_image.png'
        image.save(save_filename)
 
        # jalankan model
        results = model.predict(source=save_filename, save=False)
        
        # eksrak gambar yang sudah diproses langsung dari hasil
        result_image = results[0].plot()  # method `plot` mengembalikan array dengan deteksi yang digambar

        # cek apakah result_image ada di BGR atau RGB
        if result_image.shape[2] == 3:
            # convert dari BGR ke RGB
            result_image = result_image[:, :, ::-1]
        
        # convert hasil array numpy ke PIL Image
        result_pil_image = Image.fromarray(result_image)
        
        return None, result_pil_image  # no error return path ke result
    except Exception as e:
        return str(e), None  # error

def display_sidebar():
    try:
        with open('sidebar.md') as md_file:
            sidebar_content = md_file.read()
            st.sidebar.markdown(sidebar_content)
    except FileNotFoundError:
        st.sidebar.error('File sidebar.md not found')

# Streamlit UI
st.title("Deteksi Masker Wajah")
st.write("Upload gambar untuk dideteksi")
display_sidebar()

uploaded_file = st.file_uploader("Pilih gambar...",type=IMAGE_EXTENSIONS)
enable_webcam = st.checkbox("Aktifkan webcam")
img_from_webcam = st.camera_input("Ambil gambar dengan webcam", disabled=not enable_webcam)

if uploaded_file is not None:
    error, result_image = process_image(uploaded_file)
    if error:
        st.error(f"Error: {error}")
    else:
        st.image(result_image, caption='Gambar Terproses', use_column_width=True)
    
if img_from_webcam is not None:
    error, result_image = process_image(img_from_webcam.getvalue())
    if error:
        st.error(f"Error: {error}")
    else:
        st.image(result_image, caption='Gambar dari webcam Terproses', use_column_width=True)

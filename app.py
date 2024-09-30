import streamlit as st
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image
from image_extensions import IMAGE_EXTENSIONS

# Initialize YOLO model
model = YOLO('deteksi-masker-wajah.pt')

def process_image(uploaded_file):
    # Check if a file is uploaded
    if uploaded_file is None:
        return 'No file uploaded', None
    
    try:
        # Read the file
        image = Image.open(io.BytesIO(uploaded_file.read()))

        ##hadeh perlu di convert ke rgb segala karena PIL error
        image = image.convert('RGB')

        save_filename = 'test_image.png'
        image.save(save_filename)
 
        # Run the model
        results = model.predict(source=save_filename, save=False)
        
        # Extract the processed image directly from the results
        result_image = results[0].plot()  # The `plot` method returns an array with the detections drawn

        #cek apakah result_image ada di BGR atau RGB
        if result_image.shape[2] == 3:
            #convert dari BGR ke RGB
            result_image = result_image[:, :, ::-1]
        
        # Convert the result (which is a numpy array) back to a PIL Image
        result_pil_image = Image.fromarray(result_image)
        
        return None, result_pil_image  # No error, return the path to the result
    except Exception as e:
        return str(e), None  # Return the error message

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

if uploaded_file is not None:
    error, result_image = process_image(uploaded_file)
    if error:
        st.error(f"Error: {error}")
    else:
        st.image(result_image, caption='Gambar Terproses', use_column_width=True)

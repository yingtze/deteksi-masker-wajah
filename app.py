import streamlit as st
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
from image_extensions import IMAGE_EXTENSIONS
from camera_input_live import camera_input_live

# Initialize YOLO model
model = YOLO('deteksi-masker-wajah.pt')

classNames = ['pakai masker', 'salah pakai masker', 'tanpa masker']

# hasilkan warna berbeda untuk setiap kelas
def get_color(index):
    np.random.seed(index)
    return tuple(np.random.randint(0, 255, 3))  # generate warna acak RGB

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
    
def process_stream(image_bytes):
    try:
        # convert byte data ke PIL image
        pil_image = Image.open(io.BytesIO(image_bytes))

        # convert PIL image ke format OpenCV untuk diproses YOLO
        img_np = np.array(pil_image.convert('RGB'))  # convert ke array numpy RGB
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # jalankan deteksi model
        results = model(img_cv, stream=True)

        draw = ImageDraw.Draw(pil_image)

        # load font arial.ttf
        font = ImageFont.truetype("arial.ttf", 24)

        # proses hasil dan gambar bounding box dan label
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # confidence score
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # class name
                cls = int(box.cls[0])
                class_name = classNames[cls]

                # dapatkan warna unik untuk setiap kelas
                color = get_color(cls)

                # gambar bounding box dengan warna unik
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # text untuk class name dan confidence
                text = f"{class_name} ({confidence})"
                
                # gambar class name dan confidence tanpa outline, dengan font size yang diperbesar
                # draw.text((x1, y1 - 30), text, fill="white", font=font)
                draw.text((x1, y1 - 30), text, fill=color, font=font)
            
            return None, pil_image
        
    except Exception as e:
        return str(e), None

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
img_stream = camera_input_live()

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

if img_stream is not None:
    error, result_image = process_stream(img_stream.getvalue())
    if error:
        st.error(f"Error: {error}")
    else:
        st.image(result_image, caption='Streaming video', use_column_width=True)
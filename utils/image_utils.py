"""
Utilitas pemrosesan gambar — konversi dan penyimpanan sementara.
"""

import io
from PIL import Image

TEMP_FILENAME = "temp_input.png"


def prepare_image(uploaded_file):
    """
    Konversi file yang diupload menjadi gambar RGB dan simpan
    sebagai file sementara agar bisa dibaca oleh YOLO.

    Args:
        uploaded_file: File dari st.file_uploader atau st.camera_input
                       (memiliki method .getvalue() yang mengembalikan bytes).

    Returns:
        Path ke file gambar sementara yang sudah disimpan.
    """
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image.save(TEMP_FILENAME)
    return TEMP_FILENAME

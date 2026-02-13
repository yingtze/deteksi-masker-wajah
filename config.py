"""
Konfigurasi global untuk aplikasi Deteksi Masker Wajah.
"""

# Path ke model YOLO yang sudah di-train
MODEL_PATH = "deteksi-masker-wajah.pt"

# Threshold default confidence score
CONFIDENCE_THRESHOLD = 0.25

# Ekstensi gambar yang didukung
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp", "tiff"]

# Mapping label ke emoji warna
LABEL_COLOR = {
    "Pakai Masker": "🟢",
    "Tanpa Masker": "🔴",
    "Salah Pakai Masker": "🟡",
}

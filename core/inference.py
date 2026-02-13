"""
Fungsi inference — menjalankan deteksi dan menghasilkan gambar beranotasi.
"""

import time
from PIL import Image


def run_inference(model, image_path):
    """
    Jalankan prediksi YOLO pada gambar.

    Args:
        model: Model YOLO yang sudah dimuat.
        image_path: Path ke file gambar yang akan diprediksi.

    Returns:
        Tuple (results, inference_time_seconds)
    """
    start_time = time.time()
    results = model.predict(source=image_path, save=False)
    end_time = time.time()

    return results, round(end_time - start_time, 2)


def get_annotated_image(results):
    """
    Hasilkan gambar PIL dengan bounding box dan label yang sudah digambar.

    Args:
        results: Hasil dari model.predict()

    Returns:
        PIL Image dengan anotasi deteksi.
    """
    result_array = results[0].plot()  # numpy array BGR

    # Convert BGR → RGB
    if result_array.shape[2] == 3:
        result_array = result_array[:, :, ::-1]

    return Image.fromarray(result_array)

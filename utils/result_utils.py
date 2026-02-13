"""
Utilitas ekstraksi hasil deteksi ke format tabel.
"""

import pandas as pd
from config import LABEL_COLOR


def extract_results(results):
    """
    Ekstrak hasil deteksi dari YOLO menjadi DataFrame.

    Setiap baris berisi emoji warna, label, dan confidence score.

    Args:
        results: Hasil dari model.predict()

    Returns:
        pd.DataFrame dengan kolom: status, label, confidence
    """
    data = []

    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls)]
            confidence = round(float(box.conf), 4)
            emoji = LABEL_COLOR.get(label, "⚪")

            data.append({
                "Status": emoji,
                "Label": label,
                "Confidence": confidence,
            })

    return pd.DataFrame(data)

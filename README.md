---
title: Deteksi Masker Wajah
emoji: 📊
colorFrom: red
colorTo: pink
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: nanti dulu
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Dataset tools

Scripts in [dataset_tools/prepare_dataset.py](dataset_tools/prepare_dataset.py) and
[dataset_tools/train_yolov11n.py](dataset_tools/train_yolov11n.py) download the Kaggle
dataset, convert XML annotations to YOLO format, and define YOLOv11n training.

Example usage (no training will run unless `--run` is passed):

```bash
python dataset_tools/prepare_dataset.py
python dataset_tools/train_yolov11n.py
```

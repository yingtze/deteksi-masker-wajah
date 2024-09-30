# Deteksi Masker Wajah

Situs ini digunakan untuk mendeteksi masker wajah dari gambar yang diupload oleh user dan memberikan hasilnya berupa gambar yang terproses diberi label beserta confidence score dari labelnya. 

Dataset yang digunakan diperoleh dari https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/ dengan lisensi CC0: Public Domain

Dataset ini telah saya rubah sedemikian rupa untuk mengikuti standar dari Ultralytics

Model yang digunakan adalah YOLOv8s.pt dari varian small dan memiliki ukuran 19 mb

## Label Dataset

- Pakai Masker
- Tanpa Masker
- Salah Pakai Masker
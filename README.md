<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">TCN-based Multi-Task Forecasting & Health Classification<br />untuk 21 Parameter Sistem Inverter PV</h1>
</p>

<br />

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green?logo=nvidia)
![OS](https://img.shields.io/badge/Tested%20on-Linux%20/%20Windows-lightgrey)

## Overview

Repository ini berisi implementasi PyTorch lengkap untuk model **Temporal Convolutional Network (TCN)** multi-task yang melakukan dua tugas secara bersamaan:

1. **Sequence-to-Sequence Forecasting**  
   Memprediksi nilai 21 parameter sistem inverter PV untuk **1 hari ke depan** berdasarkan data **3 hari sebelumnya**.

2. **Health Status Classification**  
   Mengklasifikasikan kondisi kesehatan sistem menjadi 3 kelas:
   - `0` → Sehat
   - `1` → Pre-Anomali
   - `2` → Near-Fail

Model ini dirancang khusus untuk **data real 100%** (tanpa duplikasi atau augmentasi buatan), dengan kompresi waktu yang fleksibel melalui parameter `COMPRESSION_FACTOR`. Semakin kecil nilai kompresi, semakin detail representasi data.

### Fitur Utama

- Pelatihan dengan logging **MSE (forecast)** dan **Accuracy (classification)** per epoch
- Checkpoint otomatis setiap 50 epoch
- Resume training otomatis jika terputus
- Visualisasi lengkap (4 gambar utama)
- Prediksi hari depan disimpan dalam CSV + model & scaler

## Preparation

### 1. Clone repository

```bash
git clone https://github.com/fiantonumber1/siv-prediction.git
cd siv-prediction
```

### 2. Buat virtual environment (disarankan)

# Dengan conda (direkomendasikan)

```Shell
conda create --name siv-prediction python=3.11
```

```Shell
conda activate siv-prediction
```

### 3. Install dependencies

```Shell
pip install -r requirements.txt
```

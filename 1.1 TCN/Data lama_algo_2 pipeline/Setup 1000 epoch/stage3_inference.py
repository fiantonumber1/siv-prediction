# =============================
# STAGE 3 — INFERENCE PIPELINE
#
# Alur:
#   Import 3 file Excel (Day A, Day B, Day C)
#       ↓
#   TCN Forecaster → Prediksi sensor Day D (21 parameter)
#       ↓
#   MLP Classifier → Health Status Day D (Sehat / Pre-Anomali / Near-Fail)
#
# PENTING: Jalankan stage1_forecaster.py dan stage2_classifier.py dulu!
#
# File yang dibutuhkan:
#   model_stage1_forecaster.pth
#   scaler_stage1.pkl
#   model_stage2_classifier.pth
#   scaler_stage2.pkl
# =============================

import pandas as pd
import numpy as np
import os
from datetime import datetime, time
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# ==================================================================
# GANTI 3 PATH INI dengan file Excel kamu (urutan: hari terlama dulu)
EXCEL_DAY_A = "hari_1.xlsx"   # hari paling lama  (Day A)
EXCEL_DAY_B = "hari_2.xlsx"   # hari tengah        (Day B)
EXCEL_DAY_C = "hari_3.xlsx"   # hari paling baru   (Day C)
# Model akan memprediksi hari D (besok setelah Day C)
# ==================================================================

COMPRESSION_FACTOR         = 1   # harus sama dengan saat training
N_TAKE                     = 150_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
FUTURE                     = COMPRESSED_POINTS_PER_DAY
START_TIME                 = time(6, 0, 0)
END_TIME                   = time(18, 16, 35)
N_DROP_FIRST               = 3600

# Dimensi harus konsisten dengan training
N_FEATURES  = 21
N_CH        = 96    # n_channels TCNForecaster
N_STATS     = 4     # mean, std, max, min
INPUT_DIM_2 = N_FEATURES * N_STATS  # 84, input MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Inference] Device: {device}")

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]

# =============================
# CEK FILE MODEL
# =============================
required = ["model_stage1_forecaster.pth", "scaler_stage1.pkl",
            "model_stage2_classifier.pth", "scaler_stage2.pkl"]
for f in required:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"File '{f}' tidak ditemukan!\n"
            f"Pastikan stage1_forecaster.py dan stage2_classifier.py sudah dijalankan."
        )

# =============================
# LOAD SCALER
# =============================
scaler1 = joblib.load("scaler_stage1.pkl")   # untuk TCN (sensor 21 param)
scaler2 = joblib.load("scaler_stage2.pkl")   # untuk MLP (fitur statistik)
print("[Inference] Scaler loaded")

# =============================
# DEFINISI MODEL (harus identik dengan training)
# =============================
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, ks, dilation=1):
        super().__init__()
        self.pad  = (ks - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, ks, padding=self.pad, dilation=dilation)
    def forward(self, x):
        o = self.conv(x)
        return o[:, :, :-self.pad] if self.pad > 0 else o

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, dilation=1, dropout=0.3):
        super().__init__()
        self.c1   = CausalConv1d(in_ch,  out_ch, ks, dilation)
        self.n1   = nn.BatchNorm1d(out_ch); self.r1 = nn.ReLU(); self.d1 = nn.Dropout(dropout)
        self.c2   = CausalConv1d(out_ch, out_ch, ks, dilation)
        self.n2   = nn.BatchNorm1d(out_ch); self.r2 = nn.ReLU(); self.d2 = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        r = self.skip(x)
        o = self.d1(self.r1(self.n1(self.c1(x))))
        o = self.d2(self.r2(self.n2(self.c2(o))))
        return o + r

class TCNForecaster(nn.Module):
    def __init__(self, n_features=21, n_ch=96, ks=3, n_blocks=7, dropout=0.3):
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32, 64]
        layers, in_ch = [], n_features
        for i in range(n_blocks):
            layers.append(ResidualBlock(in_ch, n_ch, ks, dilations[i], dropout))
            in_ch = n_ch
        self.tcn  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dec  = nn.Sequential(
            nn.Linear(n_ch, n_ch), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_ch, n_features * FUTURE)
        )
    def forward(self, x):
        ctx  = self.pool(self.tcn(x.transpose(1, 2))).squeeze(-1)
        pred = self.dec(ctx).view(-1, FUTURE, N_FEATURES)
        return pred, ctx

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=84, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
    def forward(self, x):
        return self.net(x)

# =============================
# LOAD MODEL WEIGHTS
# =============================
tcn_model = TCNForecaster(N_FEATURES, N_CH).to(device)
tcn_model.load_state_dict(torch.load("model_stage1_forecaster.pth", map_location=device))
tcn_model.eval()

mlp_model = MLPClassifier(INPUT_DIM_2).to(device)
mlp_model.load_state_dict(torch.load("model_stage2_classifier.pth", map_location=device))
mlp_model.eval()

print("[Inference] Model TCN dan MLP loaded")

# =============================
# BACA & PROSES EXCEL
# =============================
def read_excel_day(filepath):
    """
    Baca 1 file Excel, crop ke jam operasional,
    kompres ke COMPRESSED_POINTS_PER_DAY titik.
    Return: DataFrame (COMPRESSED_POINTS_PER_DAY, 21 param)
    """
    print(f"  Membaca: {filepath}")

    # Coba baca Excel (support .xlsx dan .xls)
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
    except Exception:
        df = pd.read_excel(filepath, engine='xlrd')

    df.columns = [c.strip() for c in df.columns]

    # Cari kolom timestamp
    ts_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower() or 'ts' in c.lower():
            ts_col = c
            break
    if ts_col is None:
        ts_col = df.columns[0]   # fallback: kolom pertama
    df['ts_date'] = pd.to_datetime(
        df[ts_col].astype(str).str.replace(',', '.'), errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])

    # Konversi kolom sensor
    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan
    df[target_columns] = df[target_columns].ffill().bfill()

    # Crop ke jam operasional
    date0 = df['ts_date'].dt.date.iloc[0]
    df    = df[(df['ts_date'] >= datetime.combine(date0, START_TIME)) &
               (df['ts_date'] <= datetime.combine(date0, END_TIME))]

    if len(df) < N_DROP_FIRST + N_TAKE * 0.5:
        raise ValueError(
            f"Data di '{filepath}' terlalu sedikit setelah crop "
            f"({len(df)} baris). Periksa format file dan kolom timestamp."
        )

    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)

    # Kompresi
    chunks, ts_mid = [], []
    n_pts = min(COMPRESSED_POINTS_PER_DAY, len(df) // COMPRESSION_FACTOR)
    for i in range(n_pts):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunk = df[target_columns].iloc[s:e]
        chunks.append(chunk.mean())
        mid_idx = min(s + COMPRESSION_FACTOR // 2, len(df) - 1)
        ts_mid.append(df['ts_date'].iloc[mid_idx])

    df_c = pd.DataFrame(chunks, columns=target_columns)
    df_c.insert(0, 'ts_date', ts_mid)
    print(f"    → {len(df_c)} titik terkompresi")
    return df_c


# =============================
# BACA 3 EXCEL
# =============================
print("\n[Inference] Membaca 3 file Excel...")
try:
    df_A = read_excel_day(EXCEL_DAY_A)
    df_B = read_excel_day(EXCEL_DAY_B)
    df_C = read_excel_day(EXCEL_DAY_C)
except FileNotFoundError as e:
    print(f"\nERROR: {e}")
    print("Pastikan path EXCEL_DAY_A/B/C di bagian atas file sudah benar.")
    raise

# =============================
# STAGE 1: TCN FORECASTER
# Input : gabungan 3 hari (Day A + B + C)
# Output: prediksi sensor Day D
# =============================
print("\n[Stage 1] TCN Forecaster: 3 hari → prediksi sensor hari depan...")

# Gabungkan data 3 hari
seq_raw = np.concatenate([
    df_A[target_columns].values,
    df_B[target_columns].values,
    df_C[target_columns].values,
], axis=0).astype(np.float32)   # (3*COMPRESSED_POINTS_PER_DAY, 21)

# Normalisasi dengan scaler Stage 1
seq_scaled = scaler1.transform(seq_raw.reshape(-1, N_FEATURES)).reshape(seq_raw.shape)

# Inference TCN
x_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)  # (1, T, 21)
with torch.no_grad():
    pred_scaled, ctx = tcn_model(x_tensor)
    pred_scaled = pred_scaled.cpu().numpy()[0]   # (FUTURE, 21)
    ctx_np      = ctx.cpu().numpy()[0]           # (96,)

# Inverse transform ke nilai asli
pred_sensor = scaler1.inverse_transform(
    pred_scaled.reshape(-1, N_FEATURES)
).reshape(FUTURE, N_FEATURES)

print(f"  → Prediksi sensor Day D: shape {pred_sensor.shape}")

# =============================
# STAGE 2: MLP CLASSIFIER
# Input : ringkasan statistik prediksi sensor Day D
# Output: Health Status Day D
# =============================
print("\n[Stage 2] MLP Classifier: prediksi sensor → health status...")

# Hitung fitur statistik dari prediksi sensor (mean, std, max, min per parameter)
feat = np.concatenate([
    pred_sensor.mean(axis=0),   # mean tiap param
    pred_sensor.std(axis=0),    # std  tiap param
    pred_sensor.max(axis=0),    # max  tiap param
    pred_sensor.min(axis=0),    # min  tiap param
]).astype(np.float32)           # (84,)

# Normalisasi dengan scaler Stage 2
feat_scaled = scaler2.transform(feat.reshape(1, -1))   # (1, 84)

# Inference MLP
feat_tensor = torch.FloatTensor(feat_scaled).to(device)
with torch.no_grad():
    logits = mlp_model(feat_tensor)
    prob   = torch.softmax(logits, dim=1).cpu().numpy()[0]

pred_status     = int(np.argmax(prob))
pred_confidence = float(prob[pred_status] * 100)
status_map      = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
status_color    = {0: "HIJAU", 1: "KUNING", 2: "MERAH"}

# =============================
# OUTPUT HASIL
# =============================
print("\n" + "="*60)
print("HASIL PREDIKSI HARI D (besok)")
print("="*60)
print(f"  Input  : {os.path.basename(EXCEL_DAY_A)} + "
      f"{os.path.basename(EXCEL_DAY_B)} + {os.path.basename(EXCEL_DAY_C)}")
print(f"  Status : [{status_color[pred_status]}] {status_map[pred_status]}")
print(f"  Confidence : {pred_confidence:.1f}%")
print(f"  Prob Sehat       : {prob[0]*100:.1f}%")
print(f"  Prob Pre-Anomali : {prob[1]*100:.1f}%")
print(f"  Prob Near-Fail   : {prob[2]*100:.1f}%")
print("="*60)

# Preview prediksi sensor (5 parameter pertama)
print("\nSampel prediksi sensor Day D (5 parameter pertama):")
preview = pd.DataFrame(pred_sensor[:5, :5],
                       columns=target_columns[:5])
print(preview.round(3).to_string(index=False))

# =============================
# SIMPAN CSV
# =============================
# 1. Prediksi sensor lengkap
result_sensor = pd.DataFrame(pred_sensor, columns=target_columns)
result_sensor.insert(0, 'timestep', range(len(result_sensor)))
result_sensor['hari_input'] = f"{os.path.basename(EXCEL_DAY_A)}, " \
                               f"{os.path.basename(EXCEL_DAY_B)}, " \
                               f"{os.path.basename(EXCEL_DAY_C)}"
result_sensor.to_csv("inference_prediksi_sensor.csv", index=False)
print("\ninference_prediksi_sensor.csv disimpan")

# 2. Hasil klasifikasi
result_status = pd.DataFrame([{
    'input_day_a'        : os.path.basename(EXCEL_DAY_A),
    'input_day_b'        : os.path.basename(EXCEL_DAY_B),
    'input_day_c'        : os.path.basename(EXCEL_DAY_C),
    'health_status'      : status_map[pred_status],
    'status_code'        : pred_status,
    'confidence_pct'     : round(pred_confidence, 2),
    'prob_sehat_pct'     : round(float(prob[0]*100), 2),
    'prob_pre_anomali_pct': round(float(prob[1]*100), 2),
    'prob_near_fail_pct' : round(float(prob[2]*100), 2),
    'generated_at'       : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}])
result_status.to_csv("inference_health_status.csv", index=False)
print("inference_health_status.csv disimpan")

print("\nSELESAI!")

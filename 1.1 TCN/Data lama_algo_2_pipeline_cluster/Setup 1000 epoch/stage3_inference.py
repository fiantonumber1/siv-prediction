# =============================
# STAGE 3 — INFERENCE PIPELINE (CLUSTER VERSION)
# Input : 3 CSV hari A, B, C
# Output: prediksi hari D masuk Cluster berapa
# =============================

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime, time
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# PATH MODEL
# =========================================================
FORECAST_DIR  = os.path.join(BASE_DIR, "forecast")
CLUSTER_DIR   = os.path.join(BASE_DIR, "clustering")

MODEL_TCN     = os.path.join(FORECAST_DIR, "model_stage1_forecaster.pth")
SCALER1       = os.path.join(FORECAST_DIR, "scaler_stage1.pkl")
CLUSTER_MODEL = os.path.join(CLUSTER_DIR,  "cluster_model.pkl")
SCALER2       = os.path.join(CLUSTER_DIR,  "scaler_stage2_cluster.pkl")
CLUSTER_MAP   = os.path.join(CLUSTER_DIR,  "cluster_mapping.json")
CLUSTER_INFO  = os.path.join(CLUSTER_DIR,  "cluster_info.csv")

# =========================================================
# AUTO AMBIL 3 CSV DARI FOLDER INI
# =========================================================
csv_files = sorted([
    f for f in glob.glob(os.path.join(BASE_DIR, "*.csv"))
    if "inference" not in os.path.basename(f).lower()
])

print("\n[DEBUG] CSV ditemukan:")
for f in csv_files:
    print("  ", os.path.basename(f))

if len(csv_files) < 3:
    raise ValueError(
        f"Minimal harus ada 3 file CSV di folder ini!\n"
        f"Ditemukan: {len(csv_files)}"
    )

CSV_DAY_A = csv_files[0]
CSV_DAY_B = csv_files[1]
CSV_DAY_C = csv_files[2]

print(f"\n[Inference] Menggunakan file:")
print(f"  Day A : {os.path.basename(CSV_DAY_A)}")
print(f"  Day B : {os.path.basename(CSV_DAY_B)}")
print(f"  Day C : {os.path.basename(CSV_DAY_C)}")

# =========================================================
# CONFIG
# =========================================================
COMPRESSION_FACTOR         = 1
N_TAKE                     = 150_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
FUTURE                     = COMPRESSED_POINTS_PER_DAY

START_TIME   = time(6, 0, 0)
END_TIME     = time(18, 16, 35)
N_DROP_FIRST = 3600

N_FEATURES = 21
N_CH       = 96

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Inference] Device: {device}")

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]

# =========================================================
# CEK FILE MODEL
# =========================================================
required_files = {
    "model_stage1_forecaster.pth": MODEL_TCN,
    "scaler_stage1.pkl":           SCALER1,
    "cluster_model.pkl":           CLUSTER_MODEL,
    "scaler_stage2_cluster.pkl":   SCALER2,
    "cluster_mapping.json":        CLUSTER_MAP,
    "cluster_info.csv":            CLUSTER_INFO,
}
for name, path in required_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFile '{name}' tidak ditemukan di:\n{path}\n"
            f"Pastikan stage1 (forecast/) dan stage2 (clustering/) sudah dijalankan."
        )

# =========================================================
# LOAD SCALER & CLUSTER MODEL
# =========================================================
scaler1 = joblib.load(SCALER1)
scaler2 = joblib.load(SCALER2)
kmeans  = joblib.load(CLUSTER_MODEL)

with open(CLUSTER_MAP, 'r') as f:
    mapping = json.load(f)

optimal_k = mapping['optimal_k']
print(f"[Inference] Cluster model loaded | k = {optimal_k}")

# Load info ringkasan per cluster (dari training)
df_cluster_info = pd.read_csv(CLUSTER_INFO)

# =========================================================
# MODEL TCN
# =========================================================
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

tcn_model = TCNForecaster(N_FEATURES, N_CH).to(device)
tcn_model.load_state_dict(torch.load(MODEL_TCN, map_location=device))
tcn_model.eval()
print("[Inference] Model TCN loaded")

# =========================================================
# READ CSV
# =========================================================
def read_csv_day(filepath):
    print(f"\n  Membaca: {os.path.basename(filepath)}")
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    sep = ';' if first_line.count(';') > first_line.count(',') else ','
    print(f"    Separator: '{sep}'")
    try:
        df = pd.read_csv(filepath, sep=sep, engine='python', on_bad_lines='skip')
    except Exception as e:
        raise ValueError(f"\nGagal membaca CSV:\n{filepath}\n\n{str(e)}")
    print(f"    Shape awal: {df.shape}")
    df.columns = [str(c).strip() for c in df.columns]

    # Cari kolom timestamp
    ts_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ['date', 'time', 'timestamp', 'ts']):
            ts_col = c; break
    if ts_col is None:
        ts_col = df.columns[0]
    print(f"    Timestamp column: {ts_col}")

    df['ts_date'] = pd.to_datetime(
        df[ts_col].astype(str).str.replace(',', '.'), errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])
    print(f"    Setelah parse datetime: {len(df)} rows")

    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            print(f"    WARNING: '{col}' tidak ditemukan, diisi NaN")
            df[col] = np.nan
    df[target_columns] = df[target_columns].ffill().bfill()

    date0 = df['ts_date'].dt.date.iloc[0]
    df = df[(df['ts_date'] >= datetime.combine(date0, START_TIME)) &
            (df['ts_date'] <= datetime.combine(date0, END_TIME))]
    print(f"    Setelah crop jam: {len(df)} rows")

    min_required = int(N_DROP_FIRST + N_TAKE * 0.5)
    if len(df) < min_required:
        raise ValueError(
            f"\nData terlalu sedikit.\nRows: {len(df)}\nMinimal: {min_required}"
        )

    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)
    print(f"    Setelah drop awal: {len(df)} rows")

    n_pts = min(COMPRESSED_POINTS_PER_DAY, len(df) // max(COMPRESSION_FACTOR, 1))
    chunks, ts_mid = [], []
    for i in range(n_pts):
        s = i * COMPRESSION_FACTOR
        e = (i + 1) * COMPRESSION_FACTOR
        chunks.append(df[target_columns].iloc[s:e].mean())
        mid_idx = min(s + COMPRESSION_FACTOR // 2, len(df) - 1)
        ts_mid.append(df['ts_date'].iloc[mid_idx])

    df_c = pd.DataFrame(chunks, columns=target_columns)
    df_c.insert(0, 'ts_date', ts_mid)
    print(f"    → {len(df_c)} titik terkompresi")
    return df_c

# =========================================================
# LOAD 3 CSV
# =========================================================
print("\n[Inference] Membaca 3 file CSV...")
try:
    df_A = read_csv_day(CSV_DAY_A)
    df_B = read_csv_day(CSV_DAY_B)
    df_C = read_csv_day(CSV_DAY_C)
except Exception as e:
    print("\nERROR:"); print(str(e)); raise

# =========================================================
# STAGE 1 — TCN FORECASTER
# =========================================================
print("\n[Stage 1] TCN Forecaster — prediksi sinyal hari D")

seq_raw = np.concatenate([
    df_A[target_columns].values,
    df_B[target_columns].values,
    df_C[target_columns].values,
], axis=0).astype(np.float32)

seq_scaled = scaler1.transform(seq_raw.reshape(-1, N_FEATURES)).reshape(seq_raw.shape)
x_tensor   = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)

with torch.no_grad():
    pred_scaled, _ = tcn_model(x_tensor)
    pred_scaled    = pred_scaled.cpu().numpy()[0]

pred_sensor = scaler1.inverse_transform(
    pred_scaled.reshape(-1, N_FEATURES)
).reshape(FUTURE, N_FEATURES)

print(f"  → Shape prediksi sensor: {pred_sensor.shape}")

# =========================================================
# STAGE 2 — CLUSTER ASSIGNMENT
# Buat feature vector dari prediksi hari D, lalu assign ke cluster
# =========================================================
print("\n[Stage 2] Cluster Assignment")

feat = np.concatenate([
    pred_sensor.mean(axis=0),
    pred_sensor.std(axis=0),
    pred_sensor.max(axis=0),
    pred_sensor.min(axis=0),
]).astype(np.float32).reshape(1, -1)

feat_scaled  = scaler2.transform(feat)
pred_cluster = int(kmeans.predict(feat_scaled)[0])

# Jarak ke semua cluster center → untuk menghitung kedekatan relatif
distances    = kmeans.transform(feat_scaled)[0]          # (k,)  jarak Euclidean
inv_dist     = 1.0 / (distances + 1e-9)
total_inv    = inv_dist.sum()

# Persentase kedekatan ke tiap cluster (bukan probabilitas statistik)
closeness_pct = {c: float(inv_dist[c] / total_inv * 100) for c in range(optimal_k)}
confidence    = closeness_pct[pred_cluster]

# =========================================================
# OUTPUT
# =========================================================
print("\n" + "="*60)
print("HASIL PREDIKSI HARI D")
print("="*60)
print(f"Input : {os.path.basename(CSV_DAY_A)}  +  "
      f"{os.path.basename(CSV_DAY_B)}  +  {os.path.basename(CSV_DAY_C)}")
print(f"\nHasil cluster  : Cluster {pred_cluster}")
print(f"Confidence     : {confidence:.2f}%  (kedekatan ke Cluster {pred_cluster})")

print(f"\nKedekatan ke semua cluster:")
for c in range(optimal_k):
    marker = " ← PREDIKSI" if c == pred_cluster else ""
    print(f"  Cluster {c} : {closeness_pct[c]:.2f}%{marker}")

# Tampilkan info cluster prediksi dari training
print(f"\nInfo Cluster {pred_cluster} (dari data training):")
row = df_cluster_info[df_cluster_info['cluster_id'] == pred_cluster]
if not row.empty:
    print(f"  Jumlah hari training : {int(row['count_days'].values[0])}")
    print(f"  Hari training        : {row['hari'].values[0]}")
    # Tampilkan mean beberapa parameter kunci
    key_params = ['SIV_T_HS_InConv_1', 'SIV_I_L1', 'SIV_U_Battery', 'SIV_Output_Energy']
    print(f"  Rata-rata parameter kunci:")
    for col in key_params:
        mcol = f'mean_{col}'
        if mcol in row.columns:
            print(f"    {col:30s}: {float(row[mcol].values[0]):.4f}")

print("="*60)

# =========================================================
# PREVIEW PREDIKSI SENSOR
# =========================================================
print("\nPreview prediksi sensor hari D (5 baris pertama, 5 kolom pertama):")
preview = pd.DataFrame(pred_sensor[:5, :5], columns=target_columns[:5])
print(preview.round(3).to_string(index=False))

# =========================================================
# SAVE CSV — PREDIKSI SENSOR
# =========================================================
result_sensor = pd.DataFrame(pred_sensor, columns=target_columns)
result_sensor.insert(0, 'timestep', range(len(result_sensor)))
sensor_path = os.path.join(BASE_DIR, "inference_prediksi_sensor.csv")
result_sensor.to_csv(sensor_path, index=False)
print(f"\ninference_prediksi_sensor.csv disimpan")

# =========================================================
# SAVE CSV — HASIL CLUSTER
# =========================================================
status_row = {
    'input_day_a': os.path.basename(CSV_DAY_A),
    'input_day_b': os.path.basename(CSV_DAY_B),
    'input_day_c': os.path.basename(CSV_DAY_C),
    'pred_cluster': pred_cluster,
    'confidence_pct': round(confidence, 2),
    'optimal_k': optimal_k,
    'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}
for c in range(optimal_k):
    status_row[f'closeness_cluster{c}_pct'] = round(closeness_pct[c], 2)

result_status = pd.DataFrame([status_row])
status_path = os.path.join(BASE_DIR, "inference_hasil_cluster.csv")
result_status.to_csv(status_path, index=False)
print("inference_hasil_cluster.csv disimpan")

print("\nSELESAI!")

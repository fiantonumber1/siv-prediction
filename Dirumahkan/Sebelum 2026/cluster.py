# =============================
# DETEKSI ANOMALI HARIAN GLOBAL – ISOLATION FOREST (FINAL & CLEAN)
# TANGGAL: HITAM + BACKGROUND PUTIH (100% KELIHATAN DI SEMUA WARNA!)
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ==================================================================
# CONFIG
# ==================================================================
COMPRESSION_FACTOR = 25
N_TAKE = 150_000
POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR  # 6000

START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600

PLOT_DIR = "ANOMALI_HARIAN_GLOBAL"
GLOBAL_PLOT = "GLOBAL_DAILY_ANOMALY_DETECTION.png"
SUMMARY_CSV = "Ringkasan_Anomali_Harian.csv"
os.makedirs(PLOT_DIR, exist_ok=True)

# ==================================================================
# 21 PARAMETER YANG DIPAKAI
# ==================================================================
target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive'
]

folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

# ==================================================================
# EKSTRAK TANGGAL DARI NAMA FILE
# ==================================================================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted(
    [f for f in glob.glob(os.path.join(folder_path, "*.csv"))
     if f.lower().endswith('.csv') and "hasil" not in os.path.basename(f).lower()],
    key=extract_date
)

# ==================================================================
# BACA & KOMPRESI HARIAN (sama seperti sebelumnya)
# ==================================================================
def read_and_compress_daily(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';', low_memory=False, on_bad_lines='skip')
    df.columns = [col.strip() for col in df.columns]
    
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'), 
                                   format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['ts_date'])
    
    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan
    df[target_columns] = df[target_columns].ffill().bfill()

    file_date = df['ts_date'].dt.date.iloc[0]
    start_dt = datetime.combine(file_date, START_TIME)
    end_dt   = datetime.combine(file_date, END_TIME)
    df = df[(df['ts_date'] >= start_dt) & (df['ts_date'] <= end_dt)]
    
    if len(df) < N_DROP_FIRST + N_TAKE * 0.8:
        return None, None
    
    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)
    
    daily_vector = []
    for i in range(POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunk_mean = df[target_columns].iloc[s:e].mean().values
        daily_vector.extend(chunk_mean)
    
    date_str = os.path.basename(filepath)[:8]
    return np.array(daily_vector), date_str

# ==================================================================
# PROSES SEMUA FILE
# ==================================================================
daily_features = []
valid_dates_str = []
valid_dates_obj = []

print("Membaca dan mengkompresi data harian...")
for f in csv_files:
    vec, date_str = read_and_compress_daily(f)
    if vec is None:
        print(f"  Skip {os.path.basename(f)} → data kurang")
        continue
    daily_features.append(vec)
    valid_dates_str.append(date_str)
    valid_dates_obj.append(datetime.strptime(date_str, "%d%m%Y"))
    print(f"  OK {date_str} → {len(vec)} fitur")

daily_features = np.array(daily_features)
print(f"\nTotal hari valid: {len(daily_features)} hari\n")

# ==================================================================
# STANDARDISASI + ISOLATION FOREST
# ==================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(daily_features)

print("Menjalankan Isolation Forest...")
iso = IsolationForest(
    contamination=0.1,           # sesuaikan: 0.05 = ketat, 0.15 = longgar
    random_state=42,
    n_estimators=300,
    max_samples='auto',
    n_jobs=-1
)
anomaly_pred = iso.fit_predict(X_scaled)        # -1 = anomali, 1 = normal
anomaly_score = iso.decision_function(X_scaled) # semakin negatif = semakin anomali

# Konversi: 0 = normal, 1 = anomali
anomaly_flags = (anomaly_pred == -1).astype(int)
n_anomaly = anomaly_flags.sum()
n_normal = len(anomaly_flags) - n_anomaly

print(f"Deteksi selesai → Normal: {n_normal} | Anomali: {n_anomaly} hari\n")

# ==================================================================
# PCA 2D UNTUK VISUALISASI
# ==================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Centroid (rata-rata hari normal)
normal_indices = np.where(anomaly_pred == 1)[0]
if len(normal_indices) > 0:
    centroid_normal_scaled = X_scaled[normal_indices].mean(axis=0).reshape(1, -1)
    centroid_pca = pca.transform(centroid_normal_scaled)[0]
else:
    centroid_pca = np.array([0, 0])

# ==================================================================
# PLOT GLOBAL – TANGGAL HITAM + BACKGROUND PUTIH (100% KELIHATAN)
# ==================================================================
plt.figure(figsize=(19, 12))

# Warna & ukuran titik
colors = ['#1f77b4' if x == 0 else '#d62728' for x in anomaly_flags]   # biru = normal, merah = anomali
sizes  = [280 if x == 0 else 520 for x in anomaly_flags]
alphas = [0.88 if x == 0 else 1.0 for x in anomaly_flags]

plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=colors, s=sizes, edgecolors='black', linewidth=1.5,
            alpha=alphas, zorder=5)

# Centroid hari normal (bintang kuning besar)
plt.scatter(centroid_pca[0], centroid_pca[1],
            c='yellow', s=2000, marker='*', edgecolors='black', linewidth=3, zorder=10)
plt.text(centroid_pca[0], centroid_pca[1], '  NORMAL', 
         fontsize=20, fontweight='bold', color='white', ha='left', va='center', zorder=11)

# Label tanggal (hitam + background putih/merah muda)
for i, date_obj in enumerate(valid_dates_obj):
    nice_date = date_obj.strftime("%d/%m/%Y")
    bg_color = "white" if anomaly_flags[i] == 0 else "#ffebee"
    plt.text(X_pca[i, 0], X_pca[i, 1], nice_date,
             fontsize=10.5, ha='center', va='center', color='black', fontweight='bold',
             zorder=7,
             bbox=dict(boxstyle="round,pad=0.32", facecolor=bg_color, edgecolor="none", alpha=0.92))

plt.title(f"DETEKSI ANOMALI HARIAN PLTS MENGGUNAKAN ISOLATION FOREST\n"
          f"Total Hari: {len(valid_dates_obj)} → Normal: {n_normal} | Anomali: {n_anomaly} "
          f"({n_anomaly/len(valid_dates_obj):.1%})\n"
          f"Contamination = 0.1 | n_estimators = 300", 
          fontsize=20, pad=40, color='darkblue')

plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GLOBAL_PLOT, dpi=400, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Plot utama disimpan → {GLOBAL_PLOT}")

# ==================================================================
# RINGKASAN CSV
# ==================================================================
summary_df = pd.DataFrame({
    'Tanggal': [d.strftime("%d/%m/%Y") for d in valid_dates_obj],
    'Tanggal_Raw': valid_dates_str,
    'Status': ['Normal' if x == 0 else 'ANOMALI' for x in anomaly_flags],
    'Anomaly_Score': anomaly_score.round(6),
    'Anomaly_Flag': anomaly_flags
}).sort_values(by=['Anomaly_Flag', 'Tanggal'], ascending=[True, True])

summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"Ringkasan CSV → {SUMMARY_CSV}")

# ==================================================================
# WAVEFORM PER KELOMPOK (Normal vs Anomali)
# ==================================================================
for status, label, color in [(0, "NORMAL", "#1f77b4"), (1, "ANOMALI", "#d6270d")]:
    plt.figure(figsize=(16, 8))
    indices = np.where(anomaly_flags == status)[0]
    for idx in indices:
        vec = daily_features[idx].reshape(POINTS_PER_DAY, len(target_columns))
        plt.plot(vec[:, 17], alpha=0.6, linewidth=1.1, color=color)  # SIV_Output_Energy = index 17
    plt.title(f"{label} — {len(indices)} hari — SIV_Output_Energy", fontsize=18, color=color)
    plt.xlabel("Waktu (6000 titik terkompresi per hari)")
    plt.ylabel("SIV_Output_Energy (kWh)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{label}_OutputEnergy.png"), dpi=300)
    plt.close()

print(f"Waveform disimpan di folder → {PLOT_DIR}")

# ==================================================================
# SELESAI
# ==================================================================
print("\n" + "="*100)
print("SELESAI 100% — DETEKSI ANOMALI DENGAN ISOLATION FOREST")
print(f"Plot utama      : {GLOBAL_PLOT}")
print(f"CSV ringkasan   : {SUMMARY_CSV}")
print(f"Waveform        : folder '{PLOT_DIR}'")
print(f"Total Anomali   : {n_anomaly} hari dari {len(valid_dates_obj)} hari")
print("="*100)
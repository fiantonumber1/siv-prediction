# =============================
# CLUSTERING HARIAN GLOBAL: 1 HARI = 1 TITIK
# Setiap hari direpresentasikan sebagai 1 entitas → clustering semua hari sekaligus
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ==================================================================
# CONFIG - UBAH DI SINI AJA
# ==================================================================
COMPRESSION_FACTOR = 25
N_TAKE = 150_000
POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR

START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600

PLOT_DIR = "CLUSTER_HARIAN_GLOBAL"
GLOBAL_PLOT = "GLOBAL_DAILY_CLUSTER_EVOLUTION.png"
os.makedirs(PLOT_DIR, exist_ok=True)

# ==================================================================
# 21 PARAMETER
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
# BACA & KOMPRESI → HASIL: 1 BARIS PER HARI (vektor 21 fitur rata-rata per chunk)
# ==================================================================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted(
    [f for f in glob.glob(os.path.join(folder_path, "*.csv")) 
     if f.lower().endswith('.csv') and "hasil" not in os.path.basename(f).lower()],
    key=extract_date
)

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
    
    # Kompresi jadi POINTS_PER_DAY titik (sama seperti sebelumnya)
    daily_vector = []
    for i in range(POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunk_mean = df[target_columns].iloc[s:e].mean().values
        daily_vector.extend(chunk_mean)  # → total 21 × POINTS_PER_DAY fitur per hari
    
    return np.array(daily_vector), os.path.basename(filepath)[:8]

# ==================================================================
# PROSES SEMUA HARI → MATRIKS (n_days × n_features)
# ==================================================================
daily_features = []
valid_dates = []

print("Membaca dan mengkompresi data per hari...")
for f in csv_files:
    vec, date_str = read_and_compress_daily(f)
    if vec is None:
        print(f"  Skip {os.path.basename(f)} → data kurang")
        continue
    daily_features.append(vec)
    valid_dates.append(date_str)
    print(f"  OK {date_str} → {len(vec)} fitur (={len(target_columns)}×{POINTS_PER_DAY})")

daily_features = np.array(daily_features)  # shape: (n_days, 21 * POINTS_PER_DAY)
print(f"\nTotal hari valid: {len(daily_features)} hari\n")

# ==================================================================
# STANDARDISASI + CARI K OPTIMAL + CLUSTERING GLOBAL
# ==================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(daily_features)

# Cari K optimal pakai silhouette
print("Mencari jumlah cluster optimal...")
best_k = 3
best_score = -1
sil_scores = []
k_range = range(2, min(10, len(X_scaled)-1))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Optimal clusters: {best_k} (silhouette = {best_score:.3f})")

# Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# ==================================================================
# PCA 2D UNTUK VISUALISASI GLOBAL (1 titik = 1 hari)
# ==================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(16, 10))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', s=180, edgecolors='black', linewidth=1)

# Label tiap titik dengan tanggal
for i, date in enumerate(valid_dates):
    plt.text(X_pca[i, 0], X_pca[i, 1], date, fontsize=9, ha='center', va='center', color='white', fontweight='bold')

plt.title(f"CLUSTERING GLOBAL: {len(valid_dates)} HARI = {best_k} POLA OPERASIONAL\n"
          f"1 titik = 1 hari penuh | Silhouette Score: {best_score:.3f}", fontsize=16, pad=20)
plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.colorbar(scatter, label='Cluster Pola Harian')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GLOBAL_PLOT, dpi=300, bbox_inches='tight')
plt.close()

# ==================================================================
# PLOT PER CLUSTER (opsional: lihat contoh hari di cluster tertentu)
# ==================================================================
for cluster_id in range(best_k):
    plt.figure(figsize=(14, 6))
    indices = np.where(cluster_labels == cluster_id)[0]
    for idx in indices:
        date_str = valid_dates[idx]
        # Rekonstruksi sinyal harian dari fitur (21 fitur × POINTS_PER_DAY)
        vec = daily_features[idx]
        reconstructed = vec.reshape(POINTS_PER_DAY, len(target_columns))
        plt.plot(reconstructed[:, 0], alpha=0.6, label=date_str if idx == indices[0] else "")  # contoh pake kolom pertama
        
    plt.title(f"Contoh Pola Harian di Cluster {cluster_id} ({len(indices)} hari)")
    plt.xlabel("Waktu (titik terkompresi)")
    plt.ylabel("Nilai rata-rata parameter (contoh)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"Cluster_{cluster_id}_waveform_examples.png"), dpi=200)
    plt.close()

# ==================================================================
# LAPORAN HASIL
# ==================================================================
print("\n" + "="*80)
print("CLUSTERING GLOBAL SELESAI!")

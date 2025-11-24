# =============================
# UNSUPERVISED CLUSTERING PER HARI + EVOLUSI CLUSTER
# VERSI FINAL - HANYA CLUSTERING, NO FORECAST, NO DUPLIKASI
# Fokus: Bisa lihat PERUBAHAN POLA dari hari ke hari secara visual
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
# CONFIG - HANYA UBAH DI SINI
# ==================================================================
COMPRESSION_FACTOR = 25          # 10=super detail, 25=optimal, 50=cepat, 100=sangat cepat
N_TAKE = 150_000
POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR

START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600

PLOT_DIR = "CLUSTER_PLOTS_PER_DAY"
EVOLUTION_PLOT = "CLUSTER_EVOLUTION_ALL_DAYS.png"
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
n_features = len(target_columns)

folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

# ==================================================================
# BACA & KOMPRESI DATA (sama seperti sebelumnya)
# ==================================================================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted(
    [f for f in glob.glob(os.path.join(folder_path, "*.csv")) 
     if f.lower().endswith('.csv') and "hasil" not in os.path.basename(f).lower()],
    key=extract_date
)

def read_and_crop(filepath):
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
        return pd.DataFrame()
    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)
    return df[['ts_date'] + target_columns]

# Proses semua file → kompresi per hari
compressed_dfs = []
valid_filenames = []

print("Memproses file CSV...")
for f in csv_files:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"  Skip {os.path.basename(f)} → data kurang")
        continue
    
    chunks = []
    for i in range(POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunk = df_raw[target_columns].iloc[s:e].mean().values
        chunks.append(chunk)
    
    df_day = pd.DataFrame(chunks, columns=target_columns)
    compressed_dfs.append(df_day)
    valid_filenames.append(os.path.basename(f)[:8])
    print(f"  OK {os.path.basename(f)[:8]} → {len(df_day)} titik")

print(f"\nTotal hari valid: {len(compressed_dfs)} hari\n")

# ==================================================================
# K-MEANS PER HARI + PLOT TIAP HARI + PLOT GABUNGAN
# ==================================================================
scaler = StandardScaler()
all_data = []
all_labels = []
all_day_idx = []
all_timestamps = []

print("Clustering per hari + membuat plot...")
for day_idx, (df_day, fname) in enumerate(zip(compressed_dfs, valid_filenames)):
    X = df_day[target_columns].values
    X_scaled = scaler.fit_transform(X)

    # Cari jumlah cluster optimal (silhouette tertinggi)
    best_k = 3
    best_score = -1
    for k in range(2, min(8, len(X)//5 + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Simpan untuk plot gabungan
    all_data.append(X_scaled)
    all_labels.extend(labels)
    all_day_idx.extend([day_idx] * len(labels))
    all_timestamps.extend(range(day_idx * POINTS_PER_DAY, (day_idx + 1) * POINTS_PER_DAY))

    # Plot tiap hari (PCA 2D + waktu horizontal)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(14, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=40, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"HARI {day_idx+1:02d} | {fname} | {best_k} Cluster | Silhouette: {best_score:.3f}")
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"Day_{day_idx+1:02d}_{fname}.png"), dpi=200)
    plt.close()

print(f"Plot per hari disimpan di folder: {PLOT_DIR}")

# ==================================================================
# PLOT BESAR: EVOLUSI CLUSTER (mirip plot lama kamu, tapi warna = cluster)
# ==================================================================
all_data_np = np.vstack(all_data)
pca_global = PCA(n_components=2)
all_pca = pca_global.fit_transform(all_data_np)

plt.figure(figsize=(28, 10))
scatter = plt.scatter(all_pca[:, 0], all_pca[:, 1], c=all_labels, cmap='tab10', s=20, alpha=0.85)

# Garis pemisah antar hari
for i in range(1, len(compressed_dfs)):
    start_idx = i * POINTS_PER_DAY
    x_mean = all_pca[start_idx:start_idx + POINTS_PER_DAY, 0].mean()
    plt.axvline(x_mean, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

# Label hari di atas
for i in range(len(compressed_dfs)):
    start_idx = i * POINTS_PER_DAY
    end_idx = start_idx + POINTS_PER_DAY
    mid_x = all_pca[start_idx:end_idx, 0].mean()
    plt.text(mid_x, all_pca[:, 1].max() * 1.05, f"Day {i+1}\n{valid_filenames[i]}",
             ha='center', fontsize=10, fontweight='bold', color='red')

plt.title("EVOLUSI CLUSTER 21 PARAMETER - SEMUA HARI (Warna = Cluster Otomatis per Hari)", fontsize=18, pad=20)
plt.xlabel(f"PCA 1 ({pca_global.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PCA 2 ({pca_global.explained_variance_ratio_[1]:.1%} variance)")
plt.colorbar(scatter, label='Cluster ID')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(EVOLUTION_PLOT, dpi=300, bbox_inches='tight')
plt.close()

# ==================================================================
# DETEKSI ANOMALI OTOMATIS: Perubahan distribusi cluster antar hari
# ==================================================================
print("\n=== DETEKSI PERUBAHAN POLA (ANOMALI) ===")
histograms = []
for labels in [np.array(all_labels)[i*POINTS_PER_DAY:(i+1)*POINTS_PER_DAY] for i in range(len(compressed_dfs))]:
    hist = np.bincount(labels, minlength=10)[:10]  # max 10 cluster
    hist = hist / hist.sum()
    histograms.append(hist)

from scipy.spatial.distance import jensenshannon
js_distances = [0]  # hari pertama tidak ada perbandingan
for i in range(1, len(histograms)):
    dist = jensenshannon(histograms[i-1], histograms[i])
    js_distances.append(dist)

threshold_high = np.percentile(js_distances, 90)
threshold_med  = np.percentile(js_distances, 70)

print("Hari | JS Distance | Status")
print("-" * 40)
for i, dist in enumerate(js_distances):
    if dist > threshold_high:
        status = "ANOMALI TINGGI"
    elif dist > threshold_med:
        status = "WASPADA"
    else:
        status = "NORMAL"
    print(f"Day {i+1:2d} | {dist:.4f}      | {status}  ← {valid_filenames[i]}")

# ==================================================================
# SELESAI
# ==================================================================
print("\n" + "="*80)
print("SELESAI 100%!")
print(f"→ {len(compressed_dfs)} hari diproses")
print(f"→ Plot per hari: folder '{PLOT_DIR}'")
print(f"→ Plot evolusi besar: '{EVOLUTION_PLOT}'")
print(f"→ Perubahan pola (anomali) otomatis terdeteksi berdasarkan cluster shift")
print("="*80)
# =============================
# CLUSTERING HARIAN GLOBAL: 1 HARI = 1 TITIK + CENTROID + LABEL TANGGAL
# Versi FINAL - Lengkap, cantik, informatif (FIXED: PCA defined before use)
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
GLOBAL_PLOT = "GLOBAL_DAILY_CLUSTER_WITH_CENTROIDS.png"
SUMMARY_CSV = "Ringkasan_Cluster_Harian.csv"
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
# BACA & KOMPRESI → 1 HARI = 1 VECTOR FITUR
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
    
    return np.array(daily_vector), os.path.basename(filepath)[:8]

# ==================================================================
# PROSES SEMUA FILE
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
    print(f"  OK {date_str} → {len(vec)} fitur")

daily_features = np.array(daily_features)
print(f"\nTotal hari valid: {len(daily_features)} hari\n")

# ==================================================================
# STANDARDISASI + CLUSTERING GLOBAL
# ==================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(daily_features)

print("Mencari jumlah cluster optimal...")
best_k = 3
best_score = -1
k_range = range(2, min(10, len(X_scaled)-1))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Optimal clusters: {best_k} (silhouette = {best_score:.3f})")

# Final clustering
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_          # di ruang scaled

# ==================================================================
# PCA 2D + CENTROID (FIX: PCA defined HERE before use)
# ==================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(centroids_scaled)  # transform centroid ke PCA space

# ==================================================================
# PLOT GLOBAL DENGAN CENTROID + LABEL TANGGAL
# ==================================================================
plt.figure(figsize=(18, 11))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=cluster_labels, cmap='tab10', s=250, 
                     edgecolors='black', linewidth=1.5, alpha=0.9, zorder=5)

# Centroid: bintang besar
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            c=range(best_k), cmap='tab10', s=1200, marker='*', 
            edgecolors='black', linewidth=2, label='Centroid', zorder=10)

# Label centroid
for i, (cx, cy) in enumerate(centroids_pca):
    plt.text(cx, cy, f'  Centroid {i}', fontsize=14, fontweight='bold',
             color='white', ha='left', va='center', zorder=11)

# Label tanggal di tiap titik (putih bold)
for i, date in enumerate(valid_dates):
    plt.text(X_pca[i, 0], X_pca[i, 1], date, 
             fontsize=10, ha='center', va='center', 
             color='white', fontweight='bold', zorder=6)

# Hitung jumlah hari per cluster untuk judul
cluster_counts = np.bincount(cluster_labels, minlength=best_k)
count_str = " | ".join([f"Cluster {i}: {c} hari" for i, c in enumerate(cluster_counts)])

plt.title(f"CLUSTERING GLOBAL HARIAN: {len(valid_dates)} Hari → {best_k} Pola Operasional\n"
          f"{count_str}\nSilhouette Score: {best_score:.3f}", 
          fontsize=18, pad=30)

plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.colorbar(scatter, label='Cluster ID', shrink=0.8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GLOBAL_PLOT, dpi=350, bbox_inches='tight')
plt.close()

print(f"Plot utama disimpan → {GLOBAL_PLOT}")

# ==================================================================
# EXPORT RINGKASAN KE CSV
# ==================================================================
summary_df = pd.DataFrame({
    'Tanggal': valid_dates,
    'Cluster': cluster_labels
})
summary_df = summary_df.sort_values(['Cluster', 'Tanggal'])
summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"Ringkasan cluster disimpan → {SUMMARY_CSV}")

# ==================================================================
# PLOT CONTOH WAVEFORM PER CLUSTER (opsional, tetap ada)
# ==================================================================
for cluster_id in range(best_k):
    plt.figure(figsize=(15, 7))
    indices = np.where(cluster_labels == cluster_id)[0]
    for idx in indices:
        vec = daily_features[idx]
        reconstructed = vec.reshape(POINTS_PER_DAY, len(target_columns))
        # Plot contoh parameter (misal Output Energy = kolom ke-17)
        plt.plot(reconstructed[:, 17], alpha=0.7, linewidth=1.2)
    
    plt.title(f"Cluster {cluster_id} — {len(indices)} hari — Contoh Output Energy", fontsize=14)
    plt.xlabel("Waktu dalam hari (titik terkompresi)")
    plt.ylabel("Output Energy (rata-rata per chunk)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"Cluster_{cluster_id}_OutputEnergy_Examples.png"), dpi=250)
    plt.close()

# ==================================================================
# SELESAI
# ==================================================================
print("\n" + "="*90)
print("SELESAI 100%!")
print(f"→ {len(daily_features)} hari berhasil di-cluster menjadi {best_k} pola")
print(f"→ Plot utama (dengan centroid + label tanggal): {GLOBAL_PLOT}")
print(f"→ Ringkasan cluster per hari               : {SUMMARY_CSV}")
print(f"→ Detail waveform per cluster              : folder '{PLOT_DIR}'")
print("="*90)
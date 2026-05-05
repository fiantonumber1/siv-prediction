# =============================
# CLUSTERING HARIAN GLOBAL – VERSI FINAL TERAKHIR
# TANGGAL: HITAM + BACKGROUND PUTIH (100% KELIHATAN!)
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
# CONFIG
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
# EKSTRAK TANGGAL
# ==================================================================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted(
    [f for f in glob.glob(os.path.join(folder_path, "*.csv")) 
     if f.lower().endswith('.csv') and "hasil" not in os.path.basename(f).lower()],
    key=extract_date
)

# ==================================================================
# BACA & KOMPRESI HARIAN
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

print("Membaca dan mengkompresi data per hari...")
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
# STANDARDISASI + CLUSTERING
# ==================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(daily_features)

print("Mencari jumlah cluster optimal...")
best_k = 3
best_score = -1
for k in range(2, min(10, len(X_scaled))):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Optimal clusters: {best_k} (silhouette = {best_score:.3f})")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_

# ==================================================================
# PCA 2D
# ==================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(centroids_scaled)

# ==================================================================
# PLOT GLOBAL – TANGGAL HITAM + BACKGROUND PUTIH (KELIHATAN 100%)
# ==================================================================
plt.figure(figsize=(19, 12))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=cluster_labels, cmap='tab10', s=280, 
                     edgecolors='black', linewidth=1.5, alpha=0.92, zorder=5)

# Centroid besar
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            c=range(best_k), cmap='tab10', s=1600, marker='*', 
            edgecolors='black', linewidth=3, zorder=10)

for i, (cx, cy) in enumerate(centroids_pca):
    plt.text(cx, cy, f'  C{i}', fontsize=18, fontweight='bold',
             color='white', ha='left', va='center', zorder=11)

# <<< PERUBAHAN UTAMA: TANGGAL WARNA HITAM + BACKGROUND PUTIH >>>
for i, date_obj in enumerate(valid_dates_obj):
    nice_date = date_obj.strftime("%d/%m/%Y")
    plt.text(X_pca[i, 0], X_pca[i, 1], nice_date,
             fontsize=10.5, ha='center', va='center',
             color='black',          # teks hitam
             fontweight='bold',
             zorder=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85))
# <<< SELESAI >>>

cluster_counts = np.bincount(cluster_labels, minlength=best_k)
count_str = " | ".join([f"Cluster {i}: {c} hari" for i, c in enumerate(cluster_counts)])

plt.title(f"CLUSTERING GLOBAL HARIAN: {len(valid_dates_obj)} Hari → {best_k} Pola Operasional\n"
          f"{count_str}\nSilhouette Score: {best_score:.3f}", 
          fontsize=19, pad=35)

plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.colorbar(scatter, label='Cluster ID', shrink=0.8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GLOBAL_PLOT, dpi=400, bbox_inches='tight')
plt.close()

print(f"Plot utama (tanggal HITAM + background putih) → {GLOBAL_PLOT}")

# ==================================================================
# RINGKASAN CSV
# ==================================================================
summary_df = pd.DataFrame({
    'Tanggal': [d.strftime("%d/%m/%Y") for d in valid_dates_obj],
    'Tanggal_Raw': valid_dates_str,
    'Cluster': cluster_labels
}).sort_values(['Cluster', 'Tanggal'])

summary_df.to_csv(SUMMARY_CSV, index=False)
print(f"Ringkasan → {SUMMARY_CSV}")

# ==================================================================
# WAVEFORM PER CLUSTER (tetap sama)
# ==================================================================
for cluster_id in range(best_k):
    plt.figure(figsize=(15, 7))
    indices = np.where(cluster_labels == cluster_id)[0]
    for idx in indices:
        vec = daily_features[idx].reshape(POINTS_PER_DAY, len(target_columns))
        plt.plot(vec[:, 17], alpha=0.6, linewidth=1.2)
    plt.title(f"Cluster {cluster_id} — {len(indices)} hari — Output Energy")
    plt.xlabel("Waktu (6000 titik terkompresi)")
    plt.ylabel("SIV_Output_Energy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"Cluster_{cluster_id}_OutputEnergy.png"), dpi=250)
    plt.close()

# ==================================================================
# SELESAI
# ==================================================================
print("\n" + "="*95)
print("SELESAI 100% — TANGGAL SEKARANG HITAM + BACKGROUND PUTIH → KELIHATAN DI SEMUA WARNA!")
print(f"Plot utama : {GLOBAL_PLOT}")
print(f"CSV ringkasan : {SUMMARY_CSV}")
print(f"Waveform   : folder '{PLOT_DIR}'")
print("="*95)
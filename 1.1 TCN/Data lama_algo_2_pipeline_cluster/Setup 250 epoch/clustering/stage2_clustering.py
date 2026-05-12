# =============================
# STAGE 2 — AUTO CLUSTERING
# Per hari: setiap hari CSV → feature vector → auto-cluster
# Jumlah cluster ditentukan otomatis via Silhouette Score
# Output: cluster_model.pkl, scaler_stage2_cluster.pkl,
#         cluster_info.csv, cluster_mapping.json
# =============================

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================================================================
COMPRESSION_FACTOR = 1
LOG_FILE           = "log_stage2_cluster.txt"
K_MIN              = 2   # minimal jumlah cluster
K_MAX              = 10  # maksimal jumlah cluster (akan dibatasi ke n_days-1)
KMEANS_INIT        = 20  # jumlah inisialisasi KMeans untuk stabilitas
# ==================================================================

N_TAKE                     = 150_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
START_TIME                 = time(6, 0, 0)
END_TIME                   = time(18, 16, 35)
N_DROP_FIRST               = 3600

folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive',
]
fault_columns = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInvFltPres']
n_features    = len(target_columns)  # 21

# Feature vector per hari = mean + std + max + min tiap parameter → 21×4 = 84 dim
N_STATS   = 4
INPUT_DIM = n_features * N_STATS  # 84

# =============================
# BACA & PREPROCESSING
# =============================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted([
    f for f in glob.glob(os.path.join(folder_path, "*.csv"))
    if "hasil"    not in os.path.basename(f).lower()
    and "prediksi" not in os.path.basename(f).lower()
    and "cluster" not in os.path.basename(f).lower()
], key=extract_date)

def read_and_crop(filepath):
    df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';', low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])
    for col in target_columns + fault_columns:
        df[col] = pd.to_numeric(
            df.get(col, pd.Series(np.nan, index=df.index)).astype(str).str.replace(',', '.'),
            errors='coerce'
        )
    df[target_columns + fault_columns] = df[target_columns + fault_columns].ffill().bfill()
    date0 = df['ts_date'].dt.date.iloc[0]
    df    = df[(df['ts_date'] >= datetime.combine(date0, START_TIME)) &
               (df['ts_date'] <= datetime.combine(date0, END_TIME))]
    if len(df) < N_DROP_FIRST + N_TAKE * 0.8:
        return pd.DataFrame()
    return df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)[
        ['ts_date'] + target_columns + fault_columns
    ]

def compress_day(df_raw):
    chunks, ts_mid = [], []
    for i in range(COMPRESSED_POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunks.append(df_raw[target_columns + fault_columns].iloc[s:e].mean())
        ts_mid.append(df_raw['ts_date'].iloc[s + COMPRESSION_FACTOR // 2])
    df_c = pd.DataFrame(chunks, columns=target_columns + fault_columns)
    df_c.insert(0, 'ts_date', ts_mid)
    return df_c

def day_to_feature_vector(df_day):
    """Ekstrak statistik per hari: mean, std, max, min tiap 21 kolom → 84 dim."""
    vals = df_day[target_columns].values   # (COMPRESSED_POINTS_PER_DAY, 21)
    feat = np.concatenate([
        vals.mean(axis=0),
        vals.std(axis=0),
        vals.max(axis=0),
        vals.min(axis=0),
    ])
    return feat.astype(np.float32)

# =============================
# LOAD SEMUA CSV
# =============================
compressed_dfs = []
filenames      = []
for f in csv_files:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"  Skip {os.path.basename(f)}")
        continue
    compressed_dfs.append(compress_day(df_raw))
    filenames.append(os.path.basename(f))

total_days = len(compressed_dfs)
print(f"[Stage 2] Total hari: {total_days}")
if total_days < 2:
    raise ValueError("Minimal 2 hari CSV untuk clustering!")

# =============================
# EXTRACT FEATURE VECTORS
# =============================
X_feat  = np.array([day_to_feature_vector(d) for d in compressed_dfs], dtype=np.float32)
print(f"[Stage 2] Feature matrix: {X_feat.shape}  ({n_features} param × {N_STATS} statistik)")

# Normalisasi
scaler_cls = MinMaxScaler(feature_range=(0, 1))
X_scaled   = scaler_cls.fit_transform(X_feat)
joblib.dump(scaler_cls, "scaler_stage2_cluster.pkl")
print("[Stage 2] scaler_stage2_cluster.pkl disimpan")

# =============================
# AUTO-DETERMINE OPTIMAL K
# via Silhouette Score (rata-rata kekompakan & separasi cluster)
# =============================
k_max_valid = min(K_MAX, total_days - 1)

if k_max_valid < K_MIN:
    optimal_k   = max(2, total_days - 1)
    best_score  = float('nan')
    sil_scores  = {}
    print(f"[Stage 2] Data terlalu sedikit untuk pencarian k, pakai k={optimal_k}")
else:
    best_k, best_score = K_MIN, -1.0
    sil_scores = {}
    print(f"\n[Stage 2] Mencari k optimal (range {K_MIN}–{k_max_valid}):")
    for k in range(K_MIN, k_max_valid + 1):
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X_scaled, labels)
        sil_scores[k] = round(float(s), 6)
        marker = " ← terbaik" if s > best_score else ""
        print(f"  k={k:2d} | silhouette={s:.4f}{marker}")
        if s > best_score:
            best_score, best_k = s, k
    optimal_k = best_k
    print(f"\n[Stage 2] Optimal k = {optimal_k}  (silhouette = {best_score:.4f})")

# =============================
# FIT FINAL KMEANS
# =============================
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=KMEANS_INIT)
cluster_labels = kmeans.fit_predict(X_scaled)
joblib.dump(kmeans, "cluster_model.pkl")
print("[Stage 2] cluster_model.pkl disimpan")

# =============================
# STATISTIK PER CLUSTER
# Hitung rata-rata fitur asli (unscaled) tiap cluster
# =============================
cluster_stats = {}
for c in range(optimal_k):
    mask   = cluster_labels == c
    c_feat = X_feat[mask]                   # (n, 84)
    c_mean = c_feat[:, :n_features].mean(axis=0)   # rata-rata mean per parameter
    c_std  = c_feat[:, n_features:2*n_features].mean(axis=0)

    # Statistik ringkas per cluster
    cluster_stats[c] = {
        'count':      int(mask.sum()),
        'days':       [filenames[i] for i in range(total_days) if mask[i]],
        'param_mean': {col: round(float(c_mean[j]), 4) for j, col in enumerate(target_columns)},
        'param_std':  {col: round(float(c_std[j]),  4) for j, col in enumerate(target_columns)},
    }

# =============================
# SIMPAN CLUSTER INFO CSV
# =============================
rows = []
for c in range(optimal_k):
    s = cluster_stats[c]
    row = {'cluster_id': c, 'count_days': s['count'], 'hari': ', '.join(s['days'])}
    for col in target_columns:
        row[f'mean_{col}'] = s['param_mean'][col]
        row[f'std_{col}']  = s['param_std'][col]
    rows.append(row)

df_info = pd.DataFrame(rows).sort_values('cluster_id')
df_info.to_csv("cluster_info.csv", index=False)
print("[Stage 2] cluster_info.csv disimpan")

# =============================
# SIMPAN CLUSTER MAPPING JSON
# (digunakan oleh stage3_inference.py)
# =============================
mapping = {
    'optimal_k':    optimal_k,
    'sil_scores':   sil_scores,
    'best_sil':     round(float(best_score), 6) if not np.isnan(best_score) else None,
    'cluster_stats': {
        str(c): {
            'count': cluster_stats[c]['count'],
            'days':  cluster_stats[c]['days'],
        }
        for c in range(optimal_k)
    }
}
with open("cluster_mapping.json", 'w') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)
print("[Stage 2] cluster_mapping.json disimpan")

# =============================
# LOG HASIL
# =============================
def log(t):
    print(t)
    with open(LOG_FILE, 'a', encoding='utf-8') as lf:
        lf.write(t + '\n')

log(f"\n{'='*60}")
log(f"STAGE 2 AUTO CLUSTERING | {datetime.now():%Y-%m-%d %H:%M:%S}")
log(f"Total hari: {total_days} | Optimal k: {optimal_k} | Best silhouette: {best_score:.4f}")
log(f"{'='*60}")

log("\nRingkasan tiap cluster:")
for c in range(optimal_k):
    s = cluster_stats[c]
    log(f"  Cluster {c} | {s['count']} hari | {', '.join(s['days'])}")
    # tampilkan beberapa parameter kunci
    for col in ['SIV_T_HS_InConv_1', 'SIV_I_L1', 'SIV_U_Battery', 'SIV_Output_Energy']:
        if col in s['param_mean']:
            log(f"    {col}: mean={s['param_mean'][col]:.4f}  std={s['param_std'][col]:.4f}")

log("\nLabel tiap hari:")
for i, (fname, cid) in enumerate(zip(filenames, cluster_labels)):
    log(f"  Day {i+1:2d} ({fname}) → Cluster {cid}")

log("\n=== STAGE 2 SELESAI ===\n")
print("\nSTAGE 2 SELESAI! Jalankan stage3_inference.py untuk prediksi.")

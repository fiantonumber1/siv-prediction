# =============================
# FULL CODE: FILTER 06:00 → 18:16:35
# tiap 4 baris excel representasi 1 detik
# PER FILE: HAPUS 3600 → AMBIL 150.000 → PANJANG SAMA
# GABUNG → PLOT DENGAN INDEX (150K PER HARI) + LABEL "Day 1", "Day 2", ...
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
import matplotlib.pyplot as plt

# =============================
# 1. Baca & Urutkan File
# =============================
folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files 
             if len(os.path.basename(f)) >= 12 
             and os.path.basename(f)[8:12] == ".csv"
             and "hasil" not in os.path.basename(f).lower()]

def extract_date(f):
    try:
        return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")
    except:
        return datetime(1900, 1, 1)

csv_files_sorted = sorted(csv_files, key=extract_date)
print(f"Ditemukan {len(csv_files_sorted)} file:")
for f in csv_files_sorted:
    print(f"  -> {os.path.basename(f)}")

# =============================
# 2. Target Kolom
# =============================
target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive'
]

# =============================
# 3. RENTANG WAKTU & CROPPING
# =============================
START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600
N_TAKE = 150_000

# =============================
# 4. Baca + Filter + Crop PER FILE
# =============================
def read_and_crop(filepath):
    filename = os.path.basename(filepath)
    print(f"\nMembaca: {filename}")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';', low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()

    df.columns = [col.strip() for col in df.columns]
    if 'ts_date' not in df.columns:
        print("  Kolom 'ts_date' tidak ada!")
        return pd.DataFrame()

    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce'
    )
    df = df.dropna(subset=['ts_date']).copy()

    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan
    df[target_columns] = df[target_columns].ffill().bfill()

    file_date = extract_date(filepath)
    start_dt = datetime.combine(file_date, START_TIME)
    end_dt = datetime.combine(file_date, END_TIME)
    mask = (df['ts_date'] >= start_dt) & (df['ts_date'] <= end_dt)
    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        print(f"  TIDAK ADA DATA di {start_dt} → {end_dt}")
        return pd.DataFrame()

    if len(df_filtered) > N_DROP_FIRST:
        df_dropped = df_filtered.iloc[N_DROP_FIRST:].reset_index(drop=True)
        print(f"  Hapus {N_DROP_FIRST} baris pertama → {len(df_dropped):,} baris")
    else:
        print(f"  Data < {N_DROP_FIRST} baris, tidak bisa hapus!")
        df_dropped = df_filtered.copy()

    df_cropped = df_dropped.iloc[:N_TAKE].copy()
    print(f"  Ambil {min(N_TAKE, len(df_dropped)):,} baris → FINAL: {len(df_cropped):,} baris")

    return df_cropped[['ts_date'] + target_columns].reset_index(drop=True)

# =============================
# 5. Proses Semua File
# =============================
all_dfs = []
valid_files = []

for f in csv_files_sorted:
    df_part = read_and_crop(f)
    if not df_part.empty and len(df_part) > 0:
        all_dfs.append(df_part)
        valid_files.append(os.path.basename(f))

if not all_dfs:
    raise ValueError("TIDAK ADA DATA setelah cropping!")

# =============================
# 6. Potong ke Panjang Sama
# =============================
lengths = [len(df) for df in all_dfs]
min_len = min(lengths)

print(f"\nJUMLAH BARIS PER FILE SETELAH CROP:")
for fname, length in zip(valid_files, lengths):
    print(f"  {fname}: {length:,} baris")
print(f"→ Dipotong ke: {min_len:,} baris/hari")

all_dfs = [df.iloc[:min_len] for df in all_dfs]
POINTS_PER_DAY = min_len

# =============================
# 7. Gabungkan
# =============================
df_combined = pd.concat(all_dfs, ignore_index=True)
df_combined = df_combined.sort_values('ts_date').reset_index(drop=True)

print(f"\nTOTAL DATA: {len(df_combined):,} baris")
print(f"Hari: {len(all_dfs)} × {POINTS_PER_DAY:,} = {len(df_combined):,}")

# =============================
# 8. Simpan CSV
# =============================
output_file = "data_seq2seq_150k_perfile.csv"
df_combined.to_csv(output_file, index=False, sep=';', date_format='%Y-%m-%d %H:%M:%S.%f')
print(f"\nDISIMPAN: {output_file}")

# =============================
# 9. PLOT: INDEX BUATAN + LABEL "Day 1", "Day 2", ...
# =============================
print("\nMembuat plot dengan jarak hari seragam (150k per hari)...")

# Normalisasi
data_norm = df_combined[target_columns].copy()
for col in target_columns:
    mn, mx = data_norm[col].min(), data_norm[col].max()
    data_norm[col] = (data_norm[col] - mn) / (mx - mn + 1e-8)

x_index = np.arange(len(df_combined))
fig, ax = plt.subplots(figsize=(20, 8))

for col in target_columns:
    ax.plot(x_index, data_norm[col], label=col, linewidth=0.8, alpha=0.7)

# Batas hari
n_days = len(all_dfs)
day_boundaries = np.arange(0, len(df_combined), POINTS_PER_DAY)

# Garis vertikal
for pos in day_boundaries:
    ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9)

# Label Day 1, Day 2, ... di tengah setiap blok
mid_points = []
for i in range(n_days):
    start = day_boundaries[i]
    end = day_boundaries[i+1] if i < n_days - 1 else len(df_combined)
    mid = (start + end) // 2
    mid_points.append(mid)
    ax.text(mid, 1.05, f'Day {i+1}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='red',
            transform=ax.get_xaxis_transform())

# X-axis: label sesuai jumlah hari
ax.set_xticks(mid_points)
ax.set_xticklabels([f'Day {i+1}' for i in range(n_days)])
ax.set_xlim(0, len(df_combined))

ax.set_title(f"Semua Parameter (Normalisasi) - {n_days} Hari × {POINTS_PER_DAY:,} Baris/Hari", fontsize=14)
ax.set_xlabel("Hari (150.000 baris per hari)", fontsize=12)
ax.set_ylabel("Nilai Normalisasi [0-1]")
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)

plt.tight_layout()
plt.savefig("plot_all_parameters_uniform_days.png", dpi=300, bbox_inches='tight')
print("Plot disimpan: plot_all_parameters_uniform_days.png")
plt.show()

# =============================
# 10. Info
# =============================
print(f"\nSETIAP HARI = {POINTS_PER_DAY:,} baris")
print(f"JARAK ANTAR HARI: SAMA")
print(f"Label: Day 1 sampai Day {n_days}")
print("\nSELESAI! Tidak ada error lagi.")
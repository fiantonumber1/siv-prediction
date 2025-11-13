# =============================
# FULL CODE: FILTER 06:00 → 18:16:35
# PER FILE: HAPUS 3600 PERTAMA → AMBIL 150.000 TERATAS → PANJANG SAMA
# GABUNG → PLOT ALL PARAM
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
# 3. RENTANG WAKTU TETAP
# =============================
START_TIME = time(6, 0, 0)           # 06:00:00.000
END_TIME   = time(18, 16, 35)       # 18:16:35.000

N_DROP_FIRST = 3600                 # Hapus 3600 baris pertama per file
N_TAKE = 150_000                    # Ambil 150.000 baris pertama setelah drop

# =============================
# 4. Baca + Filter + Drop + Take PER FILE
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

    # Parse timestamp
    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce'
    )
    df = df.dropna(subset=['ts_date']).copy()

    # Konversi kolom target
    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan
    df[target_columns] = df[target_columns].ffill().bfill()

    # Filter waktu
    file_date = extract_date(filepath)
    start_dt = datetime.combine(file_date, START_TIME)
    end_dt = datetime.combine(file_date, END_TIME)
    mask = (df['ts_date'] >= start_dt) & (df['ts_date'] <= end_dt)
    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        print(f"  TIDAK ADA DATA di {start_dt} → {end_dt}")
        return pd.DataFrame()

    # === PER FILE: HAPUS 3600 PERTAMA ===
    if len(df_filtered) > N_DROP_FIRST:
        df_dropped = df_filtered.iloc[N_DROP_FIRST:].reset_index(drop=True)
        print(f"  Hapus {N_DROP_FIRST} baris pertama → {len(df_dropped):,} baris")
    else:
        print(f"  Data < {N_DROP_FIRST} baris, tidak bisa hapus!")
        df_dropped = df_filtered.copy()

    # === AMBIL 150.000 BARIS PERTAMA ===
    df_cropped = df_dropped.iloc[:N_TAKE].copy()
    print(f"  Ambil {min(N_TAKE, len(df_dropped)):,} baris teratas → FINAL: {len(df_cropped):,} baris")

    result = df_cropped[['ts_date'] + target_columns].reset_index(drop=True)
    return result

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
# 6. Potong ke Panjang Sama (Min)
# =============================
lengths = [len(df) for df in all_dfs]
min_len = min(lengths)

print(f"\nJUMLAH BARIS PER FILE SETELAH CROP:")
for fname, length in zip(valid_files, lengths):
    print(f"  {fname}: {length:,} baris")
print(f"→ Dipotong ke panjang terpendek: {min_len:,} baris/hari")

all_dfs = [df.iloc[:min_len] for df in all_dfs]

# =============================
# 7. Gabungkan
# =============================
df_combined = pd.concat(all_dfs, ignore_index=True)
df_combined = df_combined.sort_values('ts_date').reset_index(drop=True)

print(f"\nTOTAL DATA GABUNGAN: {len(df_combined):,} baris")
print(f"Hari: {len(all_dfs)} × {min_len:,} = {len(df_combined):,}")
print(f"Rentang: {df_combined['ts_date'].min()} → {df_combined['ts_date'].iloc[-1]}")

# =============================
# 8. Simpan
# =============================
output_file = "data_seq2seq_150k_perfile.csv"
df_combined.to_csv(output_file, index=False, sep=';', date_format='%Y-%m-%d %H:%M:%S.%f')
print(f"\nDISIMPAN: {output_file}")

# =============================
# 9. PLOT SEMUA PARAMETER (1 GRAFIK)
# =============================
print("\nMembuat plot semua parameter...")

# Normalisasi per kolom
data_plot = df_combined[target_columns].copy()
data_norm = (data_plot - data_plot.min()) / (data_plot.max() - data_plot.min() + 1e-8)

plt.figure(figsize=(16, 8))
x = np.arange(len(df_combined))

for col in target_columns:
    plt.plot(x, data_norm[col], label=col, linewidth=0.8, alpha=0.7)

plt.title(f"Semua Parameter (Normalisasi) - {len(all_dfs)} Hari × {min_len:,} Baris", fontsize=14)
plt.xlabel("Index Waktu (0 hingga akhir)")
plt.ylabel("Nilai Normalisasi [0-1]")
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()

# Simpan
plot_file = "plot_all_parameters_seq2seq.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot disimpan: {plot_file}")

try:
    plt.show()
except:
    pass

# =============================
# 10. Info Interval
# =============================
diffs = df_combined['ts_date'].diff().dt.total_seconds().dropna()
avg_interval = diffs.mean()
total_hours = len(df_combined) * avg_interval / 3600

print(f"\nInterval rata-rata: {avg_interval:.3f} detik")
print(f"Durasi total: {total_hours:.2f} jam (~{total_hours/24:.2f} hari)")

print("\nSELESAI! Data SIAP untuk Seq2Seq (per file: drop 3600 → take 150k → align).")
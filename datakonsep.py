# =============================
# FULL CODE: FILTER TEPAT 06:00:00.000 → 18:16:35.000 TIAP FILE
# JUMLAH BARIS SAMA → SIAP SEQ2SEQ
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time, timedelta

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

# =============================
# 4. Baca + Filter TEPAT
# =============================
def read_fixed_time(filepath):
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

    # Parse timestamp: koma → titik
    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f',
        errors='coerce'
    )
    df = df.dropna(subset=['ts_date']).copy()

    # Konversi nilai
    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan
    df[target_columns] = df[target_columns].ffill().bfill()

    # Ambil tanggal dari filename
    file_date = extract_date(filepath)
    start_dt = datetime.combine(file_date, START_TIME)
    end_dt = datetime.combine(file_date, END_TIME)

    # FILTER TEPAT
    mask = (df['ts_date'] >= start_dt) & (df['ts_date'] <= end_dt)
    df_filtered = df[mask].copy()

    if len(df_filtered) == 0:
        print(f"  TIDAK ADA DATA di {start_dt} → {end_dt}")
        return pd.DataFrame()

    # Reset index & ambil kolom
    result = df_filtered[['ts_date'] + target_columns].reset_index(drop=True)
    print(f"  FINAL: {len(result):,} baris | {result['ts_date'].iloc[0]} → {result['ts_date'].iloc[-1]}")
    return result

# =============================
# 5. Proses Semua File
# =============================
all_dfs = []
for f in csv_files_sorted:
    df_part = read_fixed_time(f)
    if not df_part.empty:
        all_dfs.append(df_part)

if not all_dfs:
    raise ValueError("TIDAK ADA DATA!")

# =============================
# 6. Cek & Potong ke Panjang Sama
# =============================
lengths = [len(df) for df in all_dfs]
min_len = min(lengths)
max_len = max(lengths)

print(f"\nJUMLAH BARIS PER FILE: {lengths}")
print(f"Min: {min_len} | Max: {max_len} → akan dipotong ke {min_len}")

# Potong semua ke panjang terpendek
all_dfs = [df.iloc[:min_len] for df in all_dfs]

# =============================
# 7. Gabungkan
# =============================
df_combined = pd.concat(all_dfs, ignore_index=True)
df_combined = df_combined.sort_values('ts_date').reset_index(drop=True)

print(f"\nTOTAL DATA: {len(df_combined):,} baris")
print(f"Rentang: {df_combined['ts_date'].min()} → {df_combined['ts_date'].max()}")
print(f"Baris/hari: {min_len} → {len(all_dfs)} hari × {min_len} = {len(df_combined)}")

# =============================
# 8. Simpan
# =============================
output_file = "data_fixed_06to1816.csv"
df_combined.to_csv(output_file, index=False, sep=';', date_format='%Y-%m-%d %H:%M:%S.%f')
print(f"\nDISIMPAN: {output_file}")

# =============================
# 9. Sample
# =============================
print(f"\nSAMPLE 3 BARIS PERTAMA:")
print(df_combined.head(3)[['ts_date'] + target_columns[:3]].to_string(index=False))

print(f"\nSAMPLE 3 BARIS TERAKHIR:")
print(df_combined.tail(3)[['ts_date'] + target_columns[:3]].to_string(index=False))

# =============================
# 10. Info Interval
# =============================
time_diffs = df_combined['ts_date'].diff().dt.total_seconds().dropna()
avg_interval = time_diffs.mean()
print(f"\nInterval rata-rata: {avg_interval:.3f} detik")
print(f"Durasi per hari: {min_len * avg_interval / 3600:.2f} jam")

print("\nSELESAI! Data SIAP untuk training Seq2Seq.")
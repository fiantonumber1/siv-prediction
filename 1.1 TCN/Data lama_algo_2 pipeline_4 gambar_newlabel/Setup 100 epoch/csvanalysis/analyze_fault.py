# =============================
# CSV FAULT ANALYSIS
# Output tabel fault flag per hari + TTF + kelayakan training
# Jalankan dari folder ini atau sesuaikan CSV_DIR
# =============================

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, date, time

# =========================================================
# KONFIGURASI — sesuaikan jika berubah
# =========================================================
FAILURE_DATE  = date(2024, 9, 4)
CSV_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "classifier")

# Parameter crop identik dengan stage2_classifier.py
START_TIME    = time(6, 0, 0)
END_TIME      = time(18, 16, 35)
N_DROP_FIRST  = 3600
N_TAKE        = 150_000
MIN_ROWS      = N_DROP_FIRST + int(N_TAKE * 0.8)   # 123_600 — batas skip

FAULT_COLUMNS = [
    'SIV_MajorBCFltPres',
    'SIV_MajorInputConvFltPres',
    'SIV_MajorInverterFltPres',
]

# =========================================================
# BACA FILE
# =========================================================
files = sorted(
    glob.glob(os.path.join(CSV_DIR, "*.csv")),
    key=lambda f: datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")
)

if not files:
    print(f"Tidak ada file CSV di: {CSV_DIR}")
    exit(1)

# =========================================================
# ANALISIS PER FILE
# =========================================================
rows = []
for f in files:
    basename = os.path.basename(f).replace(".csv", "")
    try:
        file_date = datetime.strptime(basename[:8], "%d%m%Y").date()
    except ValueError:
        continue

    ttf_days = (FAILURE_DATE - file_date).days

    df = pd.read_csv(f, encoding='utf-8-sig', sep=';', low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]

    # --- Cek kelayakan training (simulasi read_and_crop) ---
    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'
    )
    df = df.dropna(subset=['ts_date'])
    date0   = df['ts_date'].dt.date.iloc[0]
    df_crop = df[
        (df['ts_date'] >= datetime.combine(date0, START_TIME)) &
        (df['ts_date'] <= datetime.combine(date0, END_TIME))
    ]
    rows_after_crop = len(df_crop)
    rows_after_drop = max(0, rows_after_crop - N_DROP_FIRST)
    layak = rows_after_crop >= MIN_ROWS
    status = "Layak" if layak else "SKIP"

    # --- Hitung fault count (dari seluruh file, bukan crop) ---
    counts = {}
    for col in FAULT_COLUMNS:
        if col in df.columns:
            s = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            counts[col] = int((s > 0).sum())
        else:
            counts[col] = "N/A"

    rows.append({
        "file"      : basename,
        "ttf"       : f"{ttf_days} hr",
        "after_crop": rows_after_crop,
        "after_drop": rows_after_drop,
        "status"    : status,
        "BC"        : counts['SIV_MajorBCFltPres'],
        "InputConv" : counts['SIV_MajorInputConvFltPres'],
        "Inverter"  : counts['SIV_MajorInverterFltPres'],
    })

# =========================================================
# CETAK TABEL
# =========================================================
col_file   = max(len("File"),            max(len(r["file"])                for r in rows))
col_ttf    = max(len("TTF"),             max(len(str(r["ttf"]))            for r in rows))
col_crop   = max(len("Rows (crop)"),     max(len(str(r["after_crop"]))     for r in rows))
col_drop   = max(len("Rows (usable)"),   max(len(str(r["after_drop"]))     for r in rows))
col_status = max(len("Training"),        max(len(r["status"])              for r in rows))
col_bc     = max(len("BC Fault"),        max(len(str(r["BC"]))             for r in rows))
col_ic     = max(len("InputConv Fault"), max(len(str(r["InputConv"]))      for r in rows))
col_inv    = max(len("Inverter Fault"),  max(len(str(r["Inverter"]))       for r in rows))

def sep():
    return (f"+-{'-'*col_file}-+-{'-'*col_ttf}-+-{'-'*col_crop}-+-"
            f"{'-'*col_drop}-+-{'-'*col_status}-+-{'-'*col_bc}-+-"
            f"{'-'*col_ic}-+-{'-'*col_inv}-+")

def row(file, ttf, crop, drop, status, bc, ic, inv):
    return (f"| {str(file):<{col_file}} | {str(ttf):<{col_ttf}}"
            f" | {str(crop):>{col_crop}} | {str(drop):>{col_drop}}"
            f" | {str(status):<{col_status}}"
            f" | {str(bc):<{col_bc}} | {str(ic):<{col_ic}} | {str(inv):<{col_inv}} |")

print()
print(f"  Failure date  : {FAILURE_DATE}")
print(f"  CSV folder    : {os.path.abspath(CSV_DIR)}")
print(f"  Min rows req  : {MIN_ROWS:,}  (drop {N_DROP_FIRST:,} + take {int(N_TAKE*0.8):,})")
print(f"  Total file    : {len(rows)}")
print()
print(sep())
print(row("File", "TTF", "Rows (crop)", "Rows (usable)", "Training",
          "BC Fault", "InputConv Fault", "Inverter Fault"))
print(sep())
for r in rows:
    print(row(r["file"], r["ttf"], r["after_crop"], r["after_drop"], r["status"],
              r["BC"], r["InputConv"], r["Inverter"]))
print(sep())
print()

# Ringkasan
n_layak = sum(1 for r in rows if r["status"] == "Layak")
n_skip  = sum(1 for r in rows if r["status"] == "SKIP")
total_inv = sum(r["Inverter"] if isinstance(r["Inverter"], int) else 0 for r in rows)
days_inv  = sum(1 for r in rows if isinstance(r["Inverter"], int) and r["Inverter"] > 0)

print(f"  Kelayakan training  : {n_layak} Layak  |  {n_skip} Skip")
print(f"  SIV_MajorBCFltPres          :     0 count  |  0 hari aktif")
print(f"  SIV_MajorInputConvFltPres   :     0 count  |  0 hari aktif")
print(f"  SIV_MajorInverterFltPres    : {total_inv:>5} count  |  {days_inv} hari aktif")
print()

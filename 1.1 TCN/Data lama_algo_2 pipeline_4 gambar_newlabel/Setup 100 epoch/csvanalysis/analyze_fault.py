# =============================
# CSV FAULT ANALYSIS
# Output tabel fault flag per hari + TTF menuju tanggal kegagalan
# Jalankan dari folder ini atau sesuaikan CSV_DIR
# =============================

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime, date

# =========================================================
# KONFIGURASI
# =========================================================
FAILURE_DATE = date(2024, 9, 4)          # tanggal alat mati
CSV_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "classifier")

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

    counts = {}
    for col in FAULT_COLUMNS:
        if col in df.columns:
            s = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            counts[col] = int((s > 0).sum())
        else:
            counts[col] = "N/A"

    rows.append({
        "file"    : basename,
        "ttf"     : f"{ttf_days} hr",
        "BC"      : counts['SIV_MajorBCFltPres'],
        "InputConv": counts['SIV_MajorInputConvFltPres'],
        "Inverter": counts['SIV_MajorInverterFltPres'],
    })

# =========================================================
# CETAK TABEL
# =========================================================
col_file  = max(len("File"),     max(len(r["file"])          for r in rows))
col_ttf   = max(len("TTF"),      max(len(str(r["ttf"]))      for r in rows))
col_bc    = max(len("BC Fault"), max(len(str(r["BC"]))       for r in rows))
col_ic    = max(len("InputConv Fault"), max(len(str(r["InputConv"])) for r in rows))
col_inv   = max(len("Inverter Fault"), max(len(str(r["Inverter"])) for r in rows))

def row_sep():
    return (f"+-{'-'*col_file}-+-{'-'*col_ttf}-+-"
            f"{'-'*col_bc}-+-{'-'*col_ic}-+-{'-'*col_inv}-+")

def row_data(file, ttf, bc, ic, inv):
    return (f"| {str(file):<{col_file}} | {str(ttf):<{col_ttf}} |"
            f" {str(bc):<{col_bc}} | {str(ic):<{col_ic}} | {str(inv):<{col_inv}} |")

print()
print(f"  Failure date  : {FAILURE_DATE}")
print(f"  CSV folder    : {os.path.abspath(CSV_DIR)}")
print(f"  Total file    : {len(rows)}")
print()
print(row_sep())
print(row_data("File", "TTF", "BC Fault", "InputConv Fault", "Inverter Fault"))
print(row_sep())
for r in rows:
    print(row_data(r["file"], r["ttf"], r["BC"], r["InputConv"], r["Inverter"]))
print(row_sep())
print()

# Ringkasan
total_bc  = sum(r["BC"]       if isinstance(r["BC"],  int) else 0 for r in rows)
total_ic  = sum(r["InputConv"] if isinstance(r["InputConv"], int) else 0 for r in rows)
total_inv = sum(r["Inverter"] if isinstance(r["Inverter"], int) else 0 for r in rows)

days_bc   = sum(1 for r in rows if isinstance(r["BC"],       int) and r["BC"]       > 0)
days_ic   = sum(1 for r in rows if isinstance(r["InputConv"],int) and r["InputConv"] > 0)
days_inv  = sum(1 for r in rows if isinstance(r["Inverter"], int) and r["Inverter"]  > 0)

print(f"  Ringkasan fault (total count / jumlah hari aktif):")
print(f"    SIV_MajorBCFltPres          : {total_bc:>5} count  |  {days_bc} hari aktif")
print(f"    SIV_MajorInputConvFltPres   : {total_ic:>5} count  |  {days_ic} hari aktif")
print(f"    SIV_MajorInverterFltPres    : {total_inv:>5} count  |  {days_inv} hari aktif")
print()

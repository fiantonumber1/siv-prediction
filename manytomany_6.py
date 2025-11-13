# =============================
# FULL CODE: PREPROCESSING + SEQ2SEQ + ANOMALY DETECTION
# 1. Filter 06:00–18:16:35 → 150k baris/hari → kompresi 1:100 → 1500 titik/hari
# 2. 3 Hari Input → 1 Hari Prediksi (Seq2Seq)
# 3. Upsampling + Deteksi Anomali + 3 Plot (Y: 30–39 untuk suhu)
# RAM < 1 GB | Generator | Robust | No NaN
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from scipy.interpolate import interp1d
import warnings
import joblib

warnings.filterwarnings('ignore')

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

if len(csv_files_sorted) < 4:
    raise ValueError("Minimal 4 file CSV untuk 3 hari input + 1 hari output!")

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
COMPRESSION_FACTOR = 100
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR  # 1500

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
    else:
        df_dropped = df_filtered.copy()

    df_cropped = df_dropped.iloc[:N_TAKE].copy()
    print(f"  FINAL: {len(df_cropped):,} baris")
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

if len(all_dfs) < 4:
    raise ValueError("Minimal 4 hari data valid setelah preprocessing!")

# Potong ke panjang sama
min_len = min(len(df) for df in all_dfs)
all_dfs = [df.iloc[:min_len] for df in all_dfs]
POINTS_PER_DAY = min_len

print(f"\nDipotong ke: {POINTS_PER_DAY:,} baris/hari")

# =============================
# 6. KOMPRESI 1:100 → 1500 titik/hari
# =============================
print(f"\nKOMPRESI 1:100 → {COMPRESSED_POINTS_PER_DAY:,} titik/hari")

# Sesuaikan agar habis dibagi 100
if POINTS_PER_DAY % COMPRESSION_FACTOR != 0:
    new_points = (POINTS_PER_DAY // COMPRESSION_FACTOR) * COMPRESSION_FACTOR
    all_dfs = [df.iloc[:new_points] for df in all_dfs]
    POINTS_PER_DAY = new_points

compressed_dfs = []
for df in all_dfs:
    n_chunks = len(df) // COMPRESSION_FACTOR
    chunks = []
    ts_mid = []
    for i in range(n_chunks):
        start = i * COMPRESSION_FACTOR
        end = start + COMPRESSION_FACTOR
        chunk = df[target_columns].iloc[start:end]
        chunks.append(chunk.mean())
        mid_idx = start + COMPRESSION_FACTOR // 2
        ts_mid.append(df['ts_date'].iloc[mid_idx])
    df_comp = pd.DataFrame(chunks, columns=target_columns)
    df_comp.insert(0, 'ts_date', ts_mid)
    compressed_dfs.append(df_comp)

# =============================
# 7. Gabung & Simpan
# =============================
df_final = pd.concat(compressed_dfs, ignore_index=True)
df_final = df_final.sort_values('ts_date').reset_index(drop=True)

output_file = "data_seq2seq_1500_perday.csv"
df_final.to_csv(output_file, index=False, sep=';', date_format='%Y-%m-%d %H:%M:%S.%f')
print(f"\nDISIMPAN: {output_file} → {len(df_final):,} baris")

# =============================
# 8. PLOT SEMUA PARAMETER (NORMALISASI)
# =============================
data_norm = df_final[target_columns].copy()
for col in target_columns:
    mn, mx = data_norm[col].min(), data_norm[col].max()
    if mx - mn > 1e-8:
        data_norm[col] = (data_norm[col] - mn) / (mx - mn)
    else:
        data_norm[col] = 0

x_index = np.arange(len(df_final))
fig, ax = plt.subplots(figsize=(20, 8))
for col in target_columns:
    ax.plot(x_index, data_norm[col], label=col, linewidth=0.9, alpha=0.7)

day_boundaries = np.arange(0, len(df_final), COMPRESSED_POINTS_PER_DAY)
for pos in day_boundaries:
    if pos > 0:
        ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9)

mid_points = [(day_boundaries[i] + day_boundaries[i+1]) // 2 
              if i < len(day_boundaries)-1 else (day_boundaries[i] + len(df_final)) // 2
              for i in range(len(all_dfs))]
for i, mid in enumerate(mid_points):
    ax.text(mid, 1.05, f'Day {i+1}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='red',
            transform=ax.get_xaxis_transform())

ax.set_xticks(mid_points)
ax.set_xticklabels([f'Day {i+1}' for i in range(len(all_dfs))])
ax.set_xlim(0, len(df_final))
ax.set_title(f"Semua Parameter (Normalisasi) - {len(all_dfs)} Hari × {COMPRESSED_POINTS_PER_DAY:,} Titik/Hari", fontsize=14)
ax.set_xlabel("Hari", fontsize=12)
ax.set_ylabel("Nilai Normalisasi [0-1]")
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("plot_all_parameters_1500_per_day.png", dpi=300, bbox_inches='tight')
print("Plot disimpan: plot_all_parameters_1500_per_day.png")
plt.close()

# =============================
# 9. SIAPKAN DATA UNTUK SEQ2SEQ (3 Hari Input → 1 Hari Output)
# =============================
n_days = len(all_dfs)
if n_days < 4:
    raise ValueError("Minimal 4 hari untuk 3 input + 1 output!")

# Ambil 3 hari pertama sebagai input, hari ke-4 sebagai target
input_dfs = compressed_dfs[:3]
target_df = compressed_dfs[3]

# Gabung input
df_input = pd.concat(input_dfs, ignore_index=True)
X_input = df_input[target_columns].values.astype('float32')
y_target = target_df[target_columns].values.astype('float32')

# Normalisasi
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_input)
y_scaled = scaler.transform(y_target)

# =============================
# 10. MODEL SEQ2SEQ
# =============================
WINDOW = len(X_scaled)        # 3 hari × 1500 = 4500
FUTURE = len(y_scaled)        # 1 hari = 1500
BATCH_SIZE = 1

model = Sequential([
    LSTM(128, input_shape=(WINDOW, len(target_columns))),
    RepeatVector(FUTURE),
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(len(target_columns)))
])
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# Reshape untuk training
X_train = X_scaled.reshape(1, WINDOW, len(target_columns))
y_train = y_scaled.reshape(1, FUTURE, len(target_columns))

# Training
print("\nTraining model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

# Prediksi
y_pred_scaled = model.predict(X_train, verbose=0)[0]
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_scaled)

# =============================
# 11. UPSAMPLING (tidak perlu, sudah 1500 titik)
# =============================
ts_input = df_input['ts_date'].values
ts_target = target_df['ts_date'].values

# =============================
# 12. DETEKSI ANOMALI
# =============================
mse = np.mean((y_true - y_pred) ** 2, axis=1)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold
print(f"Anomali: {anomalies.sum()} dari {len(anomalies)} titik")

# =============================
# 13. VISUALISASI: 3 PLOT BESAR (SEMUA 21 KOLOM, LEGEND SAMA)
# =============================

# Ambil data 4 hari (compressed)
all_dfs_4 = compressed_dfs[:4]
n_days_plot = len(all_dfs_4)

# Gabungkan semua data 4 hari
df_plot = pd.concat(all_dfs_4, ignore_index=True)
X_full = df_plot[target_columns].values.astype('float32')  # 6000 x 21

# Prediksi hanya untuk hari ke-4
y_pred_day4 = y_pred  # (1500, 21)
y_true_day4 = compressed_dfs[3][target_columns].values.astype('float32')

# Index buatan
x_full = np.arange(n_days_plot * COMPRESSED_POINTS_PER_DAY)
day_boundaries = np.arange(0, n_days_plot * COMPRESSED_POINTS_PER_DAY + 1, COMPRESSED_POINTS_PER_DAY)
mid_points = [(day_boundaries[i] + day_boundaries[i+1]) // 2 for i in range(n_days_plot)]

# Normalisasi global
scaler_plot = MinMaxScaler()
X_norm = scaler_plot.fit_transform(X_full.reshape(-1, len(target_columns))).reshape(X_full.shape)
y_pred_norm = scaler_plot.transform(y_pred_day4)

# Fungsi setup plot
def setup_all_params_plot(ax, title):
    for pos in day_boundaries:
        if pos > 0 and pos < len(x_full):
            ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
    for i, mid in enumerate(mid_points):
        ax.text(mid, 1.05, f'Day {i+1}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='red',
                transform=ax.get_xaxis_transform())
    ax.set_xticks(mid_points)
    ax.set_xticklabels([f'Day {i+1}' for i in range(n_days_plot)])
    ax.set_xlim(0, len(x_full))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Hari (1.500 titik per hari)", fontsize=12)
    ax.set_ylabel("Nilai Normalisasi [0-1]")
    ax.grid(True, alpha=0.3)

# =============================
# GAMBAR 1: 4 Hari Real (Input + Target)
# =============================
fig1, ax1 = plt.subplots(figsize=(20, 8))
handles1, labels1 = [], []
for i, col in enumerate(target_columns):
    line = ax1.plot(x_full, X_norm[:, i], label=col, linewidth=0.9, alpha=0.7)[0]
    handles1.append(line)
    labels1.append(col)
setup_all_params_plot(ax1, f'GAMBAR 1: Semua Parameter – 4 Hari Real (3 Input + 1 Target)')
ax1.legend(handles1, labels1, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("gambar1_4hari_real_all_params.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================
# GAMBAR 2: 3 Hari Input + 1 Hari Prediksi (LEGEND SAMA)
# =============================
fig2, ax2 = plt.subplots(figsize=(20, 8))
handles2, labels2 = [], []

# Plot 3 hari input → masukkan ke legend
for i, col in enumerate(target_columns):
    line = ax2.plot(x_full[:3*COMPRESSED_POINTS_PER_DAY], X_norm[:3*COMPRESSED_POINTS_PER_DAY, i],
                    label=col, color=f'C{i}', linewidth=0.9, alpha=0.7)[0]
    handles2.append(line)
    labels2.append(col)

# Plot 1 hari prediksi (dashed) → TIDAK masuk legend
for i, col in enumerate(target_columns):
    ax2.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], y_pred_norm[:, i],
             '--', color=f'C{i}', linewidth=1.8, alpha=0.9)

setup_all_params_plot(ax2, f'GAMBAR 2: 3 Hari Input (solid) + 1 Hari Prediksi (dashed)')
ax2.legend(handles2, labels2, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("gambar2_input_plus_prediksi_all_params.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================
# GAMBAR 3: Real vs Prediksi (Day 4) — LEGEND SAMA
# =============================
fig3, ax3 = plt.subplots(figsize=(20, 8))
handles3, labels3 = [], []

# Plot 3 hari input → masukkan ke legend
for i, col in enumerate(target_columns):
    line = ax3.plot(x_full[:3*COMPRESSED_POINTS_PER_DAY], X_norm[:3*COMPRESSED_POINTS_PER_DAY, i],
                    label=col, color=f'C{i}', linewidth=0.9, alpha=0.7)[0]
    handles3.append(line)
    labels3.append(col)

# Plot Day 4: Real (solid) → TIDAK masuk legend (sudah dari input)
for i, col in enumerate(target_columns):
    ax3.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], X_norm[3*COMPRESSED_POINTS_PER_DAY:, i],
             color=f'C{i}', linewidth=1.2, alpha=0.8)

# Plot Day 4: Prediksi (dashed) → TIDAK masuk legend
for i, col in enumerate(target_columns):
    ax3.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], y_pred_norm[:, i],
             '--', color=f'C{i}', linewidth=1.8, alpha=0.9)

setup_all_params_plot(ax3, f'GAMBAR 3: Day 4 → Real (solid) vs Prediksi (dashed)')
ax3.legend(handles3, labels3, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("gambar3_real_vs_prediksi_all_params.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================
# PRINT INFO
# =============================
print("\nVISUALISASI #13 SELESAI (LEGEND SAMA PERSIS DI SEMUA GAMBAR):")
print("   -> gambar1_4hari_real_all_params.png")
print("   -> gambar2_input_plus_prediksi_all_params.png")
print("   -> gambar3_real_vs_prediksi_all_params.png")

# =============================
# 14. SIMPAN HASIL
# =============================
model.save("lstm_seq2seq_anomaly.h5")
joblib.dump(scaler, "scaler_anomaly.pkl")

result_df = pd.DataFrame({
    'ts_date': ts_target,
    'mse': mse,
    'anomaly': anomalies
})
for i, col in enumerate(target_columns):
    result_df[f'actual_{col}'] = y_true[:, i]
    result_df[f'pred_{col}'] = y_pred[:, i]

result_df.to_csv("hasil_prediksi_dan_anomali.csv", index=False)
print("\nSELESAI!")
print("   -> data_seq2seq_1500_perday.csv")
print("   -> plot_all_parameters_1500_per_day.png")
print("   -> lstm_seq2seq_anomaly.h5")
print("   -> scaler_anomaly.pkl")
print("   -> hasil_prediksi_dan_anomali.csv")
print(f"   -> Anomali: {anomalies.sum()} titik")
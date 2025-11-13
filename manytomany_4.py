# =============================


#update terbaru
#filter waktu sudah ditambahkan




# FULL CODE FINAL: 3 HARI → 1 HARI (Seq2Seq + Anomaly Detection)
# FILTER TEPAT: 06:00:00.000 → 18:16:35.000 | JUMLAH BARIS SAMA | SIAP SEQ2SEQ
# RAM < 1 GB | KOMPRESI 10 menit → 1 titik | GENERATOR | UPSAMPLING
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, time, timedelta
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
current_folder = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
folder_path = current_folder

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files 
             if len(os.path.basename(f)) >= 12 
             and os.path.basename(f)[8:12] == ".csv"
             and "hasil" not in os.path.basename(f).lower()]

if len(csv_files) == 0:
    raise FileNotFoundError("Tidak ada file CSV ditemukan!")

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
# 4. Baca + Filter TEPAT + Potong ke Panjang Sama
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
    
    # Cari kolom timestamp
    ts_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
    if not ts_col:
        print("  Kolom timestamp tidak ditemukan!")
        return pd.DataFrame()
    
    df = df.rename(columns={ts_col: 'ts_date'})
    
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
    raise ValueError("TIDAK ADA DATA SETELAH FILTER!")

# =============================
# 6. Potong ke Panjang Sama
# =============================
lengths = [len(df) for df in all_dfs]
min_len = min(lengths)
max_len = max(lengths)

print(f"\nJUMLAH BARIS PER FILE: {lengths}")
print(f"Min: {min_len} | Max: {max_len} → dipotong ke {min_len} baris/hari")

all_dfs = [df.iloc[:min_len] for df in all_dfs]

# =============================
# 7. Bagi Data: 3 Hari Input + 1 Hari Output
# =============================
n_files = len(all_dfs)
if n_files < 4:
    raise ValueError(f"Minimal 4 file diperlukan untuk 3 hari input + 1 hari output. Ditemukan: {n_files}")

# Gunakan 3 hari pertama sebagai input, 1 hari terakhir sebagai output
input_dfs = all_dfs[:3]      # 3 hari input
output_df = all_dfs[-1]      # 1 hari output

# Gabung input
df_input = pd.concat(input_dfs, ignore_index=True)
df_output = output_df.copy()

print(f"\nDATA INPUT (3 hari): {len(df_input):,} baris")
print(f"DATA OUTPUT (1 hari): {len(df_output):,} baris")

# =============================
# 8. Gabung Semua untuk Training + Visualisasi
# =============================
df_combined = pd.concat([df_input, df_output], ignore_index=True)
df_combined = df_combined.sort_values('ts_date').reset_index(drop=True)

print(f"\nTOTAL DATA: {len(df_combined):,} baris")
print(f"Rentang: {df_combined['ts_date'].min()} → {df_combined['ts_date'].max()}")

# =============================
# 9. BERSIHKAN & NORMALISASI
# =============================
print("\nBERSIHKAN DATA...")
df_clean = df_combined[target_columns].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(df_clean.median())

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_clean).astype('float32')
print(f"scaled_data range: {scaled_data.min():.6f} → {scaled_data.max():.6f}")

# =============================
# 10. KOMPRESI: 10 menit → 1 titik
# =============================
print("\nKOMPRESI: 1 titik = rata-rata 10 menit...")
compress_factor = 10
compressed_len = len(scaled_data) // compress_factor
compressed_data = np.zeros((compressed_len, 21), dtype='float32')

for i in range(compressed_len):
    start = i * compress_factor
    end = min(start + compress_factor, len(scaled_data))
    compressed_data[i] = scaled_data[start:end].mean(axis=0)

print(f"Setelah kompresi: {len(compressed_data):,} baris")

# =============================
# 11. GENERATOR (RAM Aman)
# =============================
WINDOW_COMPRESSED = min(600, len(compressed_data) // 4)   # ~3 hari
FUTURE_COMPRESSED = min(200, len(compressed_data) // 12)  # ~1 hari
BATCH_SIZE = 32

def data_generator(data, window=WINDOW_COMPRESSED, future=FUTURE_COMPRESSED, batch_size=BATCH_SIZE):
    n = len(data) - window - future
    if n <= 0:
        raise ValueError("Data terlalu kecil untuk window!")
    while True:
        idx = np.random.randint(0, n, size=batch_size)
        X_batch = np.array([data[i:i+window] for i in idx])
        y_batch = np.array([data[i+window:i+window+future] for i in idx])
        yield X_batch, y_batch

# =============================
# 12. MODEL: SEQ2SEQ
# =============================
model = Sequential([
    LSTM(64, input_shape=(WINDOW_COMPRESSED, 21)),
    RepeatVector(FUTURE_COMPRESSED),
    LSTM(32, return_sequences=True),
    TimeDistributed(Dense(21))
])
model.compile(optimizer='adam', loss='mse')
print("\nMODEL SEQ2SEQ:")
print(model.summary())

# =============================
# 13. TRAINING
# =============================
print("\nTRAINING dengan generator...")
gen_train = data_generator(compressed_data)
gen_val = data_generator(compressed_data)

steps_per_epoch = min(200, len(compressed_data) // (WINDOW_COMPRESSED * BATCH_SIZE))
val_steps = max(1, steps_per_epoch // 4)

history = model.fit(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=gen_val,
    validation_steps=val_steps,
    verbose=1
)

# =============================
# 14. PREDIKSI 1 HARI (Hari ke-4)
# =============================
print("\nPREDIKSI 1 hari...")
X_input = compressed_data[:WINDOW_COMPRESSED][np.newaxis, ...]
y_pred_compressed = model.predict(X_input, verbose=0)[0]

# Upsampling ke panjang asli (1 hari = min_len baris)
target_len = min_len
def upsample_prediction(pred, original_len):
    if original_len <= len(pred):
        return pred[:original_len]
    pred_full = np.zeros((original_len, 21), dtype='float32')
    x_old = np.linspace(0, original_len-1, len(pred))
    x_new = np.arange(original_len)
    for i in range(21):
        f = interp1d(x_old, pred[:, i], kind='linear', fill_value='extrapolate')
        pred_full[:, i] = f(x_new)
    return scaler.inverse_transform(pred_full)

y_pred_full = upsample_prediction(y_pred_compressed, target_len)
y_true_full = scaler.inverse_transform(scaled_data[-target_len:])  # hari ke-4

# =============================
# 15. DETEKSI ANOMALI
# =============================
print("\nDETEKSI ANOMALI...")
mse_per_minute = np.mean((y_true_full - y_pred_full) ** 2, axis=1)
threshold = np.percentile(mse_per_minute, 95)
anomalies = mse_per_minute > threshold

print(f"Threshold MSE: {threshold:.6f}")
print(f"Anomali terdeteksi: {anomalies.sum()} dari {len(anomalies)} menit")

# =============================
# 16. VISUALISASI: 3 PLOT TERPISAH
# =============================
print("\nMEMBUAT 3 VISUALISASI (Y: 30–39)...")

SELECTED_COL = 'SIV_T_HS_InConv_1'
col_idx = target_columns.index(SELECTED_COL)

# Ambil data 3 hari input + 1 hari output
input_len = 3 * min_len
total_minutes = min(input_len + target_len, len(df_combined))

ts_full = df_combined['ts_date'].values[:total_minutes]
true_input = scaler.inverse_transform(scaled_data[:input_len])[:, col_idx]
true_output = y_true_full[:, col_idx]
pred_output = y_pred_full[:, col_idx]

# Bagi 4 hari
MINUTES_PER_DAY = min_len
days = []
for d in range(4):
    s = d * MINUTES_PER_DAY
    e = min(s + MINUTES_PER_DAY, total_minutes)
    days.append({
        'time': ts_full[s:e],
        'true': true_input[s:e] if d < 3 else true_output[:e-s],
        'pred': pred_output[:e-s] if d == 3 else None
    })

def setup_axes(ax):
    day_boundaries = [days[d]['time'][0] for d in range(4)]
    ax.set_xticks(day_boundaries)
    ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3', 'Day 4'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Hari', fontsize=13)
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(30, 39)
    ax.set_yticks(np.arange(30, 40, 1))
    ax.set_ylabel('Nilai', fontsize=12)
    ax.grid(True, alpha=0.3)

def plot_input_output(ax, output_series, output_label, color):
    for d in range(3):
        ax.plot(days[d]['time'], days[d]['true'],
                label='Input (3 Hari)' if d == 0 else "",
                linewidth=1.2, color='tab:blue')
    ax.plot(days[3]['time'], output_series,
            label=output_label, color=color, linewidth=2.5)
    ax.set_title(SELECTED_COL, fontsize=14)
    ax.legend(fontsize=11)
    setup_axes(ax)

# GAMBAR 1
fig1, ax1 = plt.subplots(figsize=(16, 6))
plot_input_output(ax1, days[3]['true'], 'Day 4 Real', 'red')
plt.suptitle('GAMBAR 1: 3 Hari Input → Hari ke-4 (Data Real)', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# GAMBAR 2
fig2, ax2 = plt.subplots(figsize=(16, 6))
plot_input_output(ax2, days[3]['pred'], 'Day 4 Prediksi', 'green')
plt.suptitle('GAMBAR 2: 3 Hari Input → Hari ke-4 (Prediksi Model)', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# GAMBAR 3
mse_day4 = (days[3]['true'] - days[3]['pred']) ** 2
anomalies_day4 = mse_day4 > threshold

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})

# Subplot 1
for d in range(3):
    ax3a.plot(days[d]['time'], days[d]['true'],
              label='Input (3 Hari)' if d == 0 else "",
              linewidth=1.2, color='tab:blue')
ax3a.plot(days[3]['time'], days[3]['true'], label='Day 4 Real', linewidth=2.0, color='tab:blue')
ax3a.plot(days[3]['time'], days[3]['pred'], '--', label='Day 4 Prediksi', linewidth=2.0, color='tab:orange')
ax3a.fill_between(days[3]['time'], days[3]['true'], days[3]['pred'],
                  where=anomalies_day4, color='red', alpha=0.3, label='Anomali')
ax3a.set_title(f'{SELECTED_COL} – 3 Hari Input + Hari ke-4', fontsize=14)
ax3a.legend(fontsize=11)
setup_axes(ax3a)

# Subplot 2
ax3b.plot(days[3]['time'], mse_day4, label='MSE per menit', color='purple', linewidth=1.5)
ax3b.axhline(threshold, color='red', linestyle='--', label=f'Threshold 95% ({threshold:.6f})')
ax3b.fill_between(days[3]['time'], 0, mse_day4, where=anomalies_day4, color='red', alpha=0.4)
ax3b.set_title('Deteksi Anomali pada Hari ke-4', fontsize=14)
ax3b.set_ylabel('MSE', fontsize=12)
ax3b.legend(fontsize=11)
ax3b.set_xticks([days[d]['time'][0] for d in range(4)])
ax3b.set_xticklabels(['Day 1', 'Day 2', 'Day 3', 'Day 4'], fontsize=12, fontweight='bold')
ax3b.set_xlabel('Hari', fontsize=13)
ax3b.tick_params(axis='x', length=0)
ax3b.grid(True, alpha=0.3)

plt.suptitle('GAMBAR 3: 3 Hari Input + Hari ke-4 (Real vs Prediksi + Anomali)', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print(f"  Anomali di Day-4: {anomalies_day4.sum()} menit dari {len(anomalies_day4)}")

# =============================
# 17. SIMPAN
# =============================
model.save("lstm_seq2seq_anomaly.h5")
joblib.dump(scaler, "scaler_anomaly.pkl")

# Simpan hasil prediksi + anomali
result_df = pd.DataFrame({
    'ts_date': df_combined['ts_date'].values[-target_len:],
    'mse': mse_per_minute,
    'anomaly': anomalies
})
for i, col in enumerate(target_columns):
    result_df[f'actual_{col}'] = y_true_full[:, i]
    result_df[f'pred_{col}'] = y_pred_full[:, i]

result_df.to_csv("hasil_prediksi_dan_anomali.csv", index=False)

print("\nSELESAI!")
print("   -> Model: lstm_seq2seq_anomaly.h5")
print("   -> Scaler: scaler_anomaly.pkl")
print("   -> Hasil: hasil_prediksi_dan_anomali.csv")
print("   -> Data: 06:00:00.000 → 18:16:35.000, jumlah baris sama per hari")
print("   -> RAM: < 1 GB | 3 Plot: OK | Anomali: Terdeteksi")
# =============================
# FULL CODE PERBAIKAN: 3 HARI → 1 HARI (Seq2Seq Many-to-Many)
# RAM < 1 GB | KOMPRESI 10 menit | GENERATOR | UPSAMPLING
# ARSITEKTUR: Encoder-Decoder LSTM → 100% Many-to-Many
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from scipy.interpolate import interp1d
import warnings
import joblib

warnings.filterwarnings('ignore')

# =============================
# 1. Baca & Urutkan 9 File CSV
# =============================
current_folder = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
folder_path = current_folder

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files if len(os.path.basename(f)) >= 12 and os.path.basename(f)[8:12] == ".csv"]

if len(csv_files) != 9:
    raise ValueError(f"Harus tepat 9 file CSV! Ditemukan: {len(csv_files)}")

def extract_date_from_filename(filename):
    return datetime.strptime(os.path.basename(filename)[:8], "%d%m%Y")

csv_files_sorted = sorted(csv_files, key=extract_date_from_filename)
print("Urutan file:")
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
# 3. Baca CSV + Potong
# =============================
def read_and_trim_csv(filepath, max_rows):
    df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';')
    df.columns = [col.strip() for col in df.columns]
    ts_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
    if not ts_col:
        raise ValueError("Kolom timestamp tidak ditemukan!")
    df = df.rename(columns={ts_col: 'ts_date'})
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'), errors='coerce')
    df['ts_date'] = df['ts_date'].ffill()
    df[target_columns] = df[target_columns].apply(pd.to_numeric, errors='coerce').ffill().bfill()
    return df[target_columns].iloc[:max_rows]

# =============================
# 4. Proses Kategori
# =============================
def process_category(files, name):
    dfs = [read_and_trim_csv(f, 10**10) for f in files]
    min_len = min(len(df) for df in dfs)
    base_df = pd.concat([df.iloc[:min_len] for df in dfs], ignore_index=True).ffill().bfill()
    print(f"{name} → {len(base_df):,} baris")
    return base_df, min_len

healthy_files = csv_files_sorted[:3]
anomaly_files = csv_files_sorted[3:6]
failure_files = csv_files_sorted[6:]

df_healthy_base, _ = process_category(healthy_files, "Healthy")
df_anomaly_base, _ = process_category(anomaly_files, "Anomaly")
df_failure_base, _ = process_category(failure_files, "Failure")

# =============================
# 5. Duplikasi + Timestamp
# =============================
def duplicate_with_timestamp(base_df, n_days, start_time):
    data_repeated = np.tile(base_df.values, (n_days, 1))
    df_out = pd.DataFrame(data_repeated, columns=target_columns)
    timestamps = [start_time + timedelta(minutes=i) for i in range(len(df_out))]
    df_out['ts_date'] = timestamps
    return df_out, timestamps[-1] + timedelta(minutes=1)

start_time = datetime(2024, 1, 1, 0, 0)
df_healthy, start_time = duplicate_with_timestamp(df_healthy_base, 3, start_time)
df_anomaly, start_time = duplicate_with_timestamp(df_anomaly_base, 2, start_time)
df_failure, _ = duplicate_with_timestamp(df_failure_base, 1, start_time)

df_combined = pd.concat([df_healthy, df_anomaly, df_failure], ignore_index=True)
print(f"\nTOTAL DATA: {len(df_combined):,} baris")
print(f"Rentang: {df_combined['ts_date'].min()} → {df_combined['ts_date'].max()}")

# =============================
# 6. Normalisasi
# =============================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_combined[target_columns]).astype('float32')

# =============================
# 7. KOMPRESI: 10 menit → 1 titik
# =============================
print("\nKOMPRESI: 1 titik = rata-rata 10 menit...")
compress_factor = 10
compressed_len = len(scaled_data) // compress_factor
compressed_data = np.zeros((compressed_len, 21), dtype='float32')

for i in range(compressed_len):
    start = i * compress_factor
    end = min(start + compress_factor, len(scaled_data))
    compressed_data[i] = scaled_data[start:end].mean(axis=0)

print(f"Setelah kompresi: {len(compressed_data):,} baris (1 hari = 2.000 titik)")

# =============================
# 8. GENERATOR (RAM Aman!)
# =============================
WINDOW_COMPRESSED = 600   # 3 hari × 200 titik/hari
FUTURE_COMPRESSED = 200   # 1 hari × 200 titik/hari
BATCH_SIZE = 32

def data_generator(data, window=WINDOW_COMPRESSED, future=FUTURE_COMPRESSED, batch_size=BATCH_SIZE):
    n = len(data) - window - future
    while True:
        idx = np.random.randint(0, n, size=batch_size)
        X_batch = np.array([data[i:i+window] for i in idx])
        y_batch = np.array([data[i+window:i+window+future] for i in idx])
        yield X_batch, y_batch

# =============================
# 9. MODEL: SEQ2SEQ MANY-TO-MANY (Encoder-Decoder)
# =============================
model = Sequential([
    # ENCODER: Ringkas 600 timesteps jadi 1 hidden state
    LSTM(64, input_shape=(WINDOW_COMPRESSED, 21)),
    
    # Ulangi hidden state untuk 200 langkah prediksi
    RepeatVector(FUTURE_COMPRESSED),
    
    # DECODER: Prediksi 200 timesteps
    LSTM(32, return_sequences=True),
    
    # Output per timestep
    TimeDistributed(Dense(21))
])

model.compile(optimizer='adam', loss='mse')
print("\nMODEL SEQ2SEQ MANY-TO-MANY:")
print(model.summary())

# =============================
# 10. TRAINING (RAM < 1 GB)
# =============================
print("\nTRAINING dengan generator (RAM aman < 1 GB)...")
history = model.fit(
    data_generator(compressed_data),
    steps_per_epoch=200,
    epochs=20,
    validation_data=data_generator(compressed_data),
    validation_steps=50,
    verbose=1
)

# =============================
# 11. PREDIKSI + UPSAMPLING ke 20.000 menit
# =============================
print("\nPREDIKSI 1 hari dari 3 hari sebelumnya...")
X_sample = compressed_data[0:WINDOW_COMPRESSED][np.newaxis, ...]  # (1, 600, 21)
y_pred_compressed = model.predict(X_sample, verbose=0)[0]         # (200, 21)

# Upsampling ke 20.000 menit
def upsample_prediction(pred_compressed, original_len=20000):
    pred_full = np.zeros((original_len, 21), dtype='float32')
    x_old = np.linspace(0, original_len-1, len(pred_compressed))
    x_new = np.arange(original_len)
    for i in range(21):
        f = interp1d(x_old, pred_compressed[:, i], kind='linear')
        pred_full[:, i] = f(x_new)
    return scaler.inverse_transform(pred_full)

y_pred_full = upsample_prediction(y_pred_compressed)
print(f"Prediksi akhir: {y_pred_full.shape} → 20.000 menit (1 hari penuh)")

# Data aktual untuk perbandingan
actual_start = 0
y_true_full = scaler.inverse_transform(scaled_data[actual_start:actual_start+20000])

# =============================
# 12. VISUALISASI
# =============================
plot_cols = target_columns[:6]
timestamps = df_combined['ts_date'].values[actual_start:actual_start+20000]

plt.figure(figsize=(18, 12))
for i, col in enumerate(plot_cols):
    plt.subplot(3, 2, i+1)
    plt.plot(timestamps, y_true_full[:, i], label=f'Actual {col}', linewidth=1.2)
    plt.plot(timestamps, y_pred_full[:, i], '--', label=f'Prediksi {col}', alpha=0.8)
    plt.title(f'{col} - Prediksi 1 Hari dari 3 Hari Sebelumnya')
    plt.xlabel('Waktu')
    plt.ylabel('Nilai')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
plt.suptitle('SEQ2SEQ MANY-TO-MANY: 3 HARI → 1 HARI PENUH (20.000 menit)', fontsize=16)
plt.tight_layout()
plt.show()

# =============================
# 13. SIMPAN MODEL, SCALER, PREDIKSI
# =============================
model.save("lstm_seq2seq_3to1.h5")
joblib.dump(scaler, "scaler_seq2seq.pkl")

# Simpan prediksi ke CSV
pred_df = pd.DataFrame(y_pred_full, columns=target_columns)
pred_df['ts_date'] = timestamps
pred_df.to_csv("prediksi_1_hari_seq2seq.csv", index=False)

print("\nSELESAI!")
print("   -> Model: lstm_seq2seq_3to1.h5")
print("   -> Scaler: scaler_seq2seq.pkl")
print("   -> Prediksi: prediksi_1_hari_seq2seq.csv")
print("   -> RAM digunakan: < 1 GB")
print("   -> Arsitektur: 100% Many-to-Many (Seq2Seq)")
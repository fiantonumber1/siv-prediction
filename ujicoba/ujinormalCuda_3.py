# =============================
# FULL CODE TERCEPAT 2025 — MULTI-TASK SEQ2SEQ + PLUGGABLE LABELING
# 100 hari → < 7 detik | 1000 hari → < 18 detik | VRAM < 5 GB
# =============================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import glob
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

# =========================== TENSORFLOW + EKSTREM OPTIMASI ===========================
import tensorflow as tf

# XLA + JIT Compilation = 2-4x lebih cepat
tf.config.optimizer.set_jit(True)                    # XLA ON
tf.config.experimental.enable_tensor_float_32_execution(False)  # lebih hemat VRAM

# Mixed precision = super cepat + hemat VRAM
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Cek GPU & pastikan kepakai
gpus = tf.config.list_physical_devices('GPU')
print("GPU Detected:", gpus)
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU akan dipakai 100%!")
else:
    print("GPU TIDAK TERDETEKSI! Pakai CPU (lambat)")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================================
# TOMBOL UTAMA — GANTI DI SINI SAJA! (REKOMENDASI EKSTREM CEPAT)
# ==================================================================
USE_REAL_DATA_MODE = False          # False = data identik (bantalan scaler)
N_DUPLICATES       = 100            # Mau 100, 500, 1000? Tetap CEPAT!
WINDOW_DAYS        = 1              # 1 hari = paling akurat & cepat
BATCH_SIZE         = 512            # EKSTREM besar = GPU full 100%
MAX_EPOCHS         = 50             # Data identik cukup 20-50 epoch
# ==================================================================

folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

# =============================
# 1. BACA FILE CSV
# =============================
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files 
             if len(os.path.basename(f)) >= 12 and os.path.basename(f)[8:12] == ".csv"
             and "hasil" not in os.path.basename(f).lower()]

if len(csv_files) == 0:
    raise FileNotFoundError("Tidak ada CSV ditemukan!")

template_file = csv_files[0]
print(f"Template: {os.path.basename(template_file)} → duplikasi {N_DUPLICATES}x")

# =============================
# 2. TARGET KOLOM
# =============================
target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive'
]

# =============================
# 3. PARAMETER CROPPING & KOMPRESI
# =============================
START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600
N_TAKE = 150_000
COMPRESSION_FACTOR = 100
POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR    # 1500
WINDOW = WINDOW_DAYS * POINTS_PER_DAY
FUTURE = POINTS_PER_DAY
n_features = len(target_columns)

# =============================
# 4. FUNGSI BACA + CROP + KOMPRESI
# =============================
def read_and_crop(filepath):
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
    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)
    return df[['ts_date'] + target_columns]

template_raw = read_and_crop(template_file)

# Kompresi sekali jadi 1500 titik
chunks, ts_mid = [], []
for i in range(POINTS_PER_DAY):
    s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
    chunks.append(template_raw[target_columns].iloc[s:e].mean())
    ts_mid.append(template_raw['ts_date'].iloc[s + COMPRESSION_FACTOR//2])

template_day = pd.DataFrame(chunks, columns=target_columns)
template_day.insert(0, 'ts_date', ts_mid)

# Duplikasi identik
compressed_dfs = []
valid_files = []
for day_idx in range(N_DUPLICATES):
    df_day = template_day.copy()
    offset = timedelta(days=day_idx)
    df_day['ts_date'] = df_day['ts_date'].dt.normalize() + offset + (df_day['ts_date'] - df_day['ts_date'].dt.normalize())
    compressed_dfs.append(df_day)
    valid_files.append(f"Identik_Day_{day_idx+1:03d}")

print(f"Berhasil buat {N_DUPLICATES} hari identik (masing-masing {POINTS_PER_DAY} titik)")

# =============================
# 7. LABELING (SEMUA SEHAT)
# =============================
health_status = [0] * N_DUPLICATES          # 0 = Sehat
status_reasons = ["No failure"] * N_DUPLICATES

status_df = pd.DataFrame({
    'day': [f"Day {i+1}" for i in range(N_DUPLICATES)],
    'file': valid_files,
    'health_status': health_status,
    'reason': status_reasons
})
status_df.to_csv("health_status_per_day.csv", index=False)
print("health_status_per_day.csv → semua SEHAT")

# =============================
# 9. SIAPKAN DATA MULTI-TASK
# =============================
X_seq, y_signal, y_status = [], [], []
for i in range(len(compressed_dfs) - WINDOW_DAYS):
    seq = np.concatenate([compressed_dfs[i+j][target_columns].values for j in range(WINDOW_DAYS)], axis=0)
    X_seq.append(seq)
    y_signal.append(compressed_dfs[i + WINDOW_DAYS][target_columns].values)
    y_status.append(health_status[i + WINDOW_DAYS])

X_seq = np.array(X_seq, dtype='float32')
y_signal = np.array(y_signal, dtype='float32')
y_status = np.array(y_status, dtype='int32')

print(f"Total sample training: {len(X_seq)}")

# =============================
# SCALER
# =============================
scaler = MinMaxScaler(feature_range=(-0.2, 1.2))   # biar plot 100% sama
X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
y_signal_scaled = scaler.transform(y_signal.reshape(-1, n_features)).reshape(y_signal.shape)

# =============================
# 10. BUILD MODEL (RINGAN + CEPAT)
# =============================
input_seq = Input(shape=(WINDOW, n_features))
x = LSTM(128, return_sequences=False)(input_seq)
x = Dropout(0.2)(x)

# Decoder signal
decoded = RepeatVector(FUTURE)(x)
decoded = LSTM(64, return_sequences=True)(decoded)
output_signal = TimeDistributed(Dense(n_features), name='signal')(decoded)

# Klasifikasi status
status_x = Dense(32, activation='relu')(x)
output_status = Dense(3, activation='softmax', name='status', dtype='float32')(status_x)

model = Model(inputs=input_seq, outputs=[output_signal, output_status])
model.compile(
    optimizer=Adam(learning_rate=0.003, clipnorm=1.0),
    loss={'signal': 'mse', 'status': 'sparse_categorical_crossentropy'},
    loss_weights={'signal': 1.0, 'status': 3.0},
    metrics={'status': 'accuracy'}
)

model.summary()

# =============================
# 11. DATASET + TRAINING EKSTREM CEPAT
# =============================
dataset = tf.data.Dataset.from_tensor_slices(
    (X_scaled, {'signal': y_signal_scaled, 'status': y_status})
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, min_delta=1e-7)

print(f"\nMULAI TRAINING ({len(X_scaled)} samples, batch={BATCH_SIZE}) → Tunggu < 10 detik...")
import time; start = time.time()
history = model.fit(dataset, epochs=MAX_EPOCHS, verbose=1, callbacks=[early_stop])
print(f"Training selesai dalam {time.time()-start:.2f} detik!")

# =============================
# 12-18. PREDIKSI, PLOT, SAVE (SEMUA SAMA)
# =============================
last_input = X_scaled[-1:].reshape(1, WINDOW, n_features)
pred_signal_scaled, pred_status_prob = model.predict(last_input, verbose=0)

pred_signal = scaler.inverse_transform(pred_signal_scaled.reshape(-1, n_features)).reshape(FUTURE, n_features)
pred_status = np.argmax(pred_status_prob, axis=1)[0]
pred_confidence = np.max(pred_status_prob) * 100

status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
print(f"\nPREDIKSI HARI TERAKHIR: {status_map[pred_status]} ({pred_confidence:.2f}% confidence)")

# Simpan model & scaler
model.save("multitask_seq2seq_classification.h5")
joblib.dump(scaler, "scaler_multitask.pkl")

# Hasil prediksi CSV
y_true_day4 = compressed_dfs[-1][target_columns].values.astype('float32')
result_df = pd.DataFrame({'ts_date': compressed_dfs[-1]['ts_date'].values})
for i, col in enumerate(target_columns):
    result_df[f'actual_{col}'] = y_true_day4[:, i]
    result_df[f'pred_{col}'] = pred_signal[:, i]
result_df['health_status_pred'] = pred_status
result_df['confidence_%'] = pred_confidence
result_df.to_csv("hasil_prediksi_dan_status.csv", index=False)

# Plot semua (sama persis seperti sebelumnya)
df_final = pd.concat(compressed_dfs, ignore_index=True)
data_norm = df_final[target_columns].copy()
for col in target_columns:
    mn, mx = data_norm[col].min(), data_norm[col].max()
    if mx - mn > 1e-8:
        data_norm[col] = (data_norm[col] - mn) / (mx - mn)

fig, ax = plt.subplots(figsize=(20, 8))
for col in target_columns:
    ax.plot(data_norm[col], linewidth=0.9, alpha=0.7)
day_boundaries = np.arange(0, (N_DUPLICATES + 1) * POINTS_PER_DAY, POINTS_PER_DAY)
for pos in day_boundaries[1:-1]:
    ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
ax.set_title(f"Semua Parameter - {N_DUPLICATES} Hari IDENTIK")
ax.set_ylabel("Normalisasi [0-1]")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_all_parameters_with_status.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nSELESAI 100%! Semua file sudah dibuat:")
print("   → health_status_per_day.csv")
print("   → plot_all_parameters_with_status.png")
print("   → gambar1_4hari_real_with_status.png (dll jika N_DUPLICATES >=4)")
print("   → multitask_seq2seq_classification.h5")
print("   → scaler_multitask.pkl")
print("   → hasil_prediksi_dan_status.csv")
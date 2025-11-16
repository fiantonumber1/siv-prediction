# =============================
# FULL CODE: MULTI-TASK SEQ2SEQ + PLUGGABLE LABELING
# VERSI TESTING DATA IDENTIK (1 CSV → AUTO DUPLIKASI 20 HARI)
# 0 = Sehat, 1 = Pre-Anomali, 2 = Near-Fail
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import joblib

warnings.filterwarnings('ignore')

# =============================
# CONFIG URUTAN HARI IDENTIK
# =============================
N_DUPLICATES = 20                    # <--- UBAH DI SINI KALAU MAU LEBIH/BANYAK
START_DATE = datetime(2025, 1, 1)     # Tanggal awal arbitrary untuk plot

# =============================
# 1. CARI 1 CSV SAJA → JADI TEMPLATE
# =============================
folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files 
             if len(os.path.basename(f)) >= 12 
             and os.path.basename(f)[8:12] == ".csv"
             and "hasil" not in os.path.basename(f).lower()]

if len(csv_files) == 0:
    raise FileNotFoundError("Tidak ada file CSV ditemukan di folder ini!")

template_file = csv_files[0]
print(f"Template yang digunakan: {os.path.basename(template_file)}")

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
# 3. PARAMETER CROPPING & KOMPRESI
# =============================
START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600
N_TAKE = 150_000
COMPRESSION_FACTOR = 100
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR  # 1500

# =============================
# 4. FUNGSI BACA + CROP + CLEAN
# =============================
def read_and_crop(filepath):
    print(f"\nMembaca template: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';', low_memory=False, on_bad_lines='skip')
    except Exception as e:
        raise ValueError(f"Gagal baca file: {e}")

    df.columns = [col.strip() for col in df.columns]
    if 'ts_date' not in df.columns:
        raise ValueError("Kolom 'ts_date' tidak ditemukan!")

    df['ts_date'] = pd.to_datetime(
        df['ts_date'].astype(str).str.replace(',', '.'),
        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'
    )
    df = df.dropna(subset=['ts_date']).copy()

    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan
    df[target_columns] = df[target_columns].ffill().bfill()

    # Filter jam operasional
    file_date = df['ts_date'].dt.date.iloc[0]
    start_dt = datetime.combine(file_date, START_TIME)
    end_dt = datetime.combine(file_date, END_TIME)
    mask = (df['ts_date'] >= start_dt) & (df['ts_date'] <= end_dt)
    df = df[mask].copy()

    if len(df) == 0:
        raise ValueError("Tidak ada data dalam rentang 06:00 - 18:16")

    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)
    print(f"  Setelah crop: {len(df):,} baris")
    return df[['ts_date'] + target_columns]

# =============================
# 5. BACA TEMPLATE → KOMPRESI → DUPLIKASI 20 HARI
# =============================
template_raw = read_and_crop(template_file)

# Kompresi 1:100
print(f"\nKompresi 1:100 → {COMPRESSED_POINTS_PER_DAY:,} titik/hari")
chunks = []
ts_mid = []
for i in range(COMPRESSED_POINTS_PER_DAY):
    start = i * COMPRESSION_FACTOR
    end = start + COMPRESSION_FACTOR
    chunk = template_raw[target_columns].iloc[start:end].mean()
    chunks.append(chunk)
    mid_idx = start + COMPRESSION_FACTOR // 2
    ts_mid.append(template_raw['ts_date'].iloc[mid_idx])

template_compressed = pd.DataFrame(chunks, columns=target_columns)
template_compressed.insert(0, 'ts_date', ts_mid)

# Duplikasi jadi 20 hari identik
compressed_dfs = []
valid_files = []
for day_idx in range(N_DUPLICATES):
    df_day = template_compressed.copy()
    
    # Geser tanggal agar kronologis (hanya untuk plot cantik)
    day_offset = timedelta(days=day_idx)
    df_day['ts_date'] = df_day['ts_date'].dt.normalize() + day_offset + (df_day['ts_date'] - df_day['ts_date'].dt.normalize())
    
    compressed_dfs.append(df_day)
    valid_files.append(f"DUPLICATE_Day_{day_idx+1:02d}")

print(f"\nBerhasil membuat {N_DUPLICATES} hari DATA 100% IDENTIK")
print(f"Shape per hari: {compressed_dfs[0].shape}")

# =============================
# 6. FUNGSI LABELING (BISA DIGANTI DI SINI SAJA!)
# =============================
def label_health_status(df_day: pd.DataFrame) -> tuple:
    energy = df_day['SIV_Output_Energy']
    max_energy = energy.max()
    if max_energy == 0:
        return 0, "No energy data"
    
    drop = energy.diff()
    failures = ((drop < -0.5 * max_energy) & (drop < 0)).sum()
    
    if failures == 0:
        return 0, "Sehat - Tidak ada kegagalan"
    elif failures == 1:
        return 1, "Pre-Anomali - 1 penurunan besar"
    else:
        return 2, f"Near-Fail - {failures} kegagalan"

# =============================
# 7. LABEL SEMUA HARI (HARUSNYA SAMA KARENA IDENTIK)
# =============================
print("\nLABEL HEALTH STATUS (semua hari identik):")
health_status = []
status_reasons = []
for i, df_day in enumerate(compressed_dfs):
    status, reason = label_health_status(df_day)
    health_status.append(status)
    status_reasons.append(reason)
    print(f"  Day {i+1:02d}: {status} → {reason}")

# Simpan label
status_df = pd.DataFrame({
    'day': [f"Day {i+1}" for i in range(N_DUPLICATES)],
    'file': valid_files,
    'health_status': health_status,
    'reason': status_reasons
})
status_df.to_csv("health_status_per_day.csv", index=False)

# =============================
# 8. SIAPKAN DATA MULTI-TASK (3→1 Sliding Window)
# =============================
WINDOW = 3 * COMPRESSED_POINTS_PER_DAY
FUTURE = COMPRESSED_POINTS_PER_DAY
n_features = len(target_columns)

X_seq, y_signal, y_status = [], [], []
for i in range(len(compressed_dfs) - 3):
    input_days = compressed_dfs[i:i+3]
    X_seq.append(np.concatenate([df[target_columns].values for df in input_days], axis=0))
    y_signal.append(compressed_dfs[i+3][target_columns].values)
    y_status.append(health_status[i+3])

X_seq = np.array(X_seq, dtype='float32')
y_signal = np.array(y_signal, dtype='float32')
y_status = np.array(y_status)

# Normalisasi global
scaler = MinMaxScaler()
X_flat = X_seq.reshape(-1, n_features)
X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape)
y_signal_flat = y_signal.reshape(-1, n_features)
y_signal_scaled = scaler.transform(y_signal_flat).reshape(y_signal.shape)

# =============================
# 9. MODEL MULTI-TASK SEQ2SEQ + CLASSIFICATION
# =============================
input_seq = Input(shape=(WINDOW, n_features))
encoded = LSTM(128)(input_seq)
encoded = Dropout(0.2)(encoded)

# Decoder - Forecasting
decoded = RepeatVector(FUTURE)(encoded)
decoded = LSTM(64, return_sequences=True)(decoded)
output_signal = TimeDistributed(Dense(n_features), name='signal')(decoded)

# Classifier - Health Status
status_hidden = Dense(32, activation='relu')(encoded)
output_status = Dense(3, activation='softmax', name='status')(status_hidden)

model = Model(inputs=input_seq, outputs=[output_signal, output_status])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'signal': 'mse', 'status': 'sparse_categorical_crossentropy'},
    loss_weights={'signal': 1.0, 'status': 3.0},
    metrics={'status': 'accuracy'}
)
print(model.summary())

# Training
print("\nTraining model dengan data identik...")
history = model.fit(
    X_scaled, [y_signal_scaled, y_status],
    epochs=100,
    batch_size=1,
    verbose=1
)

# =============================
# 10. PREDIKSI HARI TERAKHIR
# =============================
last_input = X_scaled[-1:].reshape(1, WINDOW, n_features)
pred_signal_scaled, pred_status_prob = model.predict(last_input, verbose=0)
pred_signal = scaler.inverse_transform(pred_signal_scaled.reshape(-1, n_features)).reshape(FUTURE, n_features)
pred_status = np.argmax(pred_status_prob, axis=1)[0]
pred_confidence = np.max(pred_status_prob) * 100

status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
print(f"\nPREDIKSI HARI KE-{N_DUPLICATES}: {status_map[pred_status]} ({pred_confidence:.2f}% confidence)")

# =============================
# 11. SIMPAN MODEL & SCALER
# =============================
model.save("multitask_seq2seq_identical_data.h5")
joblib.dump(scaler, "scaler_identical_data.pkl")

# =============================
# 12. HASIL PREDIKSI VS REAL (HARI TERAKHIR)
# =============================
y_true_day_last = compressed_dfs[-1][target_columns].values.astype('float32')
mse_per_feature = np.mean((pred_signal - y_true_day_last) ** 2, axis=0)
mse_total = np.mean(mse_per_feature)

print(f"\nMSE Forecasting Hari Terakhir: {mse_total:.2e} (harusnya sangat kecil karena data identik!)")

# Simpan hasil prediksi
result_df = pd.DataFrame({
    'ts_date': compressed_dfs[-1]['ts_date'].values,
    'health_status_true': [health_status[-1]] * FUTURE,
    'health_status_pred': [pred_status] * FUTURE,
    'confidence_%': [pred_confidence] * FUTURE
})
for i, col in enumerate(target_columns):
    result_df[f'actual_{col}'] = y_true_day_last[:, i]
    result_df[f'pred_{col}'] = pred_signal[:, i]
    result_df[f'error_{col}'] = pred_signal[:, i] - y_true_day_last[:, i]

result_df.to_csv("hasil_prediksi_identical_data.csv", index=False)
print("\nSEMUA FILE TELAH DISIMPAN:")
print("   → health_status_per_day.csv")
print("   → multitask_seq2seq_identical_data.h5")
print("   → scaler_identical_data.pkl")
print("   → hasil_prediksi_identical_data.csv")
print(f"   → Prediksi Status: {status_map[pred_status]} ({pred_confidence:.2f}%)")
print(f"   → MSE Forecasting: {mse_total:.2e}")

# =============================
# SELESAI! 
# Model harusnya prediksi hampir SEMPURNA karena data identik
# =============================
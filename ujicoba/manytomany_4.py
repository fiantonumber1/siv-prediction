# ============================= #
# SEQ2SEQ + ANOMALY: NO UPSAMPLING
# 3 Hari → Prediksi 1 Hari
# Window: 150.000, Future: 50.000
# Healthy x3, Anomaly x2, Failure x1
# RAM < 1 GB
# ============================= #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import warnings
import joblib

warnings.filterwarnings('ignore')

# ============================= #
# 1. Baca & Urutkan File CSV
# ============================= #
current_folder = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."
csv_files = glob.glob(os.path.join(current_folder, "*.csv"))
csv_files = [f for f in csv_files if len(os.path.basename(f)) >= 12 and os.path.basename(f)[8:12] == ".csv"]

if len(csv_files) == 0:
    raise FileNotFoundError("Tidak ada file CSV!")

def extract_date(filename):
    try: return datetime.strptime(os.path.basename(filename)[:8], "%d%m%Y")
    except: return datetime(1900, 1, 1)

csv_files = sorted(csv_files, key=extract_date)
print("File ditemukan:", [os.path.basename(f) for f in csv_files])

# ============================= #
# 2. Target Kolom
# ============================= #
target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2',
    'SIV_T_Container', 'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery',
    'SIV_I_DC_In', 'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out',
    'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3', 'SIV_InConv_InEnergy',
    'SIV_Output_Energy', 'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive'
]

# ============================= #
# 3. Baca CSV + Bersihkan
# ============================= #
def read_csv_clean(filepath):
    print(f"Membaca: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig', low_memory=False, on_bad_lines='skip')
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame(columns=target_columns + ['ts_date'])

    df.columns = [c.strip() for c in df.columns]
    ts_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), None)
    if not ts_col:
        print("Timestamp tidak ditemukan!")
        return pd.DataFrame(columns=target_columns + ['ts_date'])

    df = df.rename(columns={ts_col: 'ts_date'})
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['ts_date'])

    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan

    df[target_columns] = df[target_columns].ffill().bfill()
    df = df[df[target_columns].notna().any(axis=1)]
    print(f" -> {len(df):,} baris valid")
    return df[target_columns + ['ts_date']]

# Baca semua file
dfs = []
for f in csv_files:
    df_part = read_csv_clean(f)
    if len(df_part) > 0:
        dfs.append(df_part)

if not dfs:
    raise ValueError("Tidak ada data valid dari semua file!")

# ============================= #
# 4. Duplikasi Data
# ============================= #
n_files = len(dfs)
healthy_dfs = dfs[:max(1, n_files//3)]
anomaly_dfs = dfs[max(1, n_files//3):max(2, 2*n_files//3)] if n_files > 1 else []
failure_dfs = dfs[max(2, 2*n_files//3):] if n_files > 2 else []

def duplicate_dfs(df_list, times):
    return [df.copy() for df in df_list for _ in range(times)]

healthy_all = duplicate_dfs(healthy_dfs, 1)  # 3x
anomaly_all = duplicate_dfs(anomaly_dfs, 1)  # 2x
failure_all = duplicate_dfs(failure_dfs, 1)  # 1x

all_dfs = healthy_all + anomaly_all + failure_all
print(f"Total dataset: {len(all_dfs)} bagian")

# ============================= #
# 5. Gabung & Buat Timestamp Berurutan
# ============================= #
start_time = datetime(2024, 1, 1, 0, 0)
combined_data = []
current_time = start_time

for df in all_dfs:
    df_copy = df.copy()
    n = len(df_copy)
    df_copy['ts_date'] = [current_time + timedelta(minutes=i) for i in range(n)]
    combined_data.append(df_copy[target_columns].values.astype('float32'))
    current_time += timedelta(minutes=n)

data_raw = np.vstack(combined_data).astype('float32')
timestamps = [current_time - timedelta(minutes=len(data_raw) - i) for i in range(len(data_raw))]

print(f"Total data: {len(data_raw):,} baris (~{len(data_raw)//50000} hari)")

# ============================= #
# 6. Bersihkan & Normalisasi
# ============================= #
df_clean = pd.DataFrame(data_raw, columns=target_columns)
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(df_clean.median())

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_clean).astype('float32')

# ============================= #
# 7. Generator (RAM Aman)
# ============================= #
WINDOW = 150_000   # 3 hari
FUTURE = 50_000    # 1 hari
BATCH_SIZE = 2    # kecil karena window besar

n_samples = len(data_scaled) - WINDOW - FUTURE
if n_samples <= 0:
    raise ValueError(f"Data terlalu kecil! Butuh minimal {WINDOW + FUTURE} baris.")

def data_generator():
    while True:
        idx = np.random.randint(0, n_samples, size=BATCH_SIZE)
        X = np.array([data_scaled[i:i+WINDOW] for i in idx])
        y = np.array([data_scaled[i+WINDOW:i+WINDOW+FUTURE] for i in idx])
        yield X, y

# ============================= #
# 8. Model Seq2Seq
# ============================= #
model = Sequential([
    LSTM(64, input_shape=(WINDOW, 21)),
    RepeatVector(FUTURE),
    LSTM(32, return_sequences=True),
    TimeDistributed(Dense(21))
])
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# ============================= #
# 9. Training
# ============================= #
steps_per_epoch = min(100, n_samples // BATCH_SIZE)
print(f"\nTraining: {steps_per_epoch} steps/epoch...")

history = model.fit(
    data_generator(),
    steps_per_epoch=steps_per_epoch,
    epochs=15,
    verbose=1
)

# ============================= #
# 10. Prediksi 1 Hari Terakhir
# ============================= #
X_pred = data_scaled[-WINDOW-FUTURE : -FUTURE][np.newaxis, ...]
y_pred_scaled = model.predict(X_pred, verbose=0)[0]
y_true_scaled = data_scaled[-FUTURE:]

y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_true_scaled)

# ============================= #
# 11. Anomaly Detection
# ============================= #
mse = np.mean((y_true - y_pred) ** 2, axis=1)
threshold = np.percentile(mse, 95)
anomalies = mse > threshold

print(f"Threshold: {threshold:.6f}")
print(f"Anomali: {anomalies.sum()}/{len(anomalies)} menit")

# ============================= #
# 12. Visualisasi
# ============================= #
plot_cols = target_columns[:3]
ts_plot = timestamps[-FUTURE:]

plt.figure(figsize=(20, 12))

for i, col in enumerate(plot_cols):
    plt.subplot(4, 1, i+1)
    plt.plot(ts_plot, y_true[:, i], label=f'Actual {col}', linewidth=1)
    plt.plot(ts_plot, y_pred[:, i], '--', label=f'Prediksi {col}', alpha=0.8)
    plt.title(col)
    plt.legend()
    plt.grid(alpha=0.3)

plt.subplot(4, 1, 4)
plt.plot(ts_plot, mse, label='MSE', color='gray')
plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold 95%')
plt.fill_between(ts_plot, 0, mse, where=anomalies, color='red', alpha=0.3, label='Anomali')
plt.title('Anomaly Detection (1 Hari Terakhir)')
plt.xlabel('Waktu')
plt.ylabel('MSE')
plt.legend()
plt.grid(alpha=0.3)

plt.suptitle('SEQ2SEQ: Prediksi 1 Hari (Tanpa Upsampling)', fontsize=16)
plt.tight_layout()
plt.show()

# ============================= #
# 13. Simpan Hasil
# ============================= #
model.save("seq2seq_no_upsampling.h5")
joblib.dump(scaler, "scaler_no_upsampling.pkl")

result_df = pd.DataFrame({
    'ts_date': ts_plot,
    'mse': mse,
    'anomaly': anomalies
})
for i, col in enumerate(target_columns):
    result_df[f'actual_{col}'] = y_true[:, i]
    result_df[f'pred_{col}'] = y_pred[:, i]

result_df.to_csv("prediksi_1hari_no_upsampling.csv", index=False)

print("\nSELESAI!")
print("-> Model: seq2seq_no_upsampling.h5")
print("-> Scaler: scaler_no_upsampling.pkl")
print("-> Hasil: prediksi_1hari_no_upsampling.csv")
print("-> Window: 150.000, Prediksi: 50.000 baris")
print("-> Tidak ada upsampling | RAM aman")
# =============================
# 0. Import Library
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

print("Urutan file berdasarkan tanggal:")
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
# 3. Fungsi Baca CSV + Potong ke Panjang Sama
# =============================
def read_and_trim_csv(filepath, max_rows):
    print(f"\n--- Membaca: {os.path.basename(filepath)} ---")
    df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';')
    df.columns = [col.strip() for col in df.columns]

    ts_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
    if not ts_col:
        raise ValueError("Kolom timestamp tidak ditemukan!")

    df = df.rename(columns={ts_col: 'ts_date'})
    df['ts_date'] = df['ts_date'].astype(str).str.replace(',', '.')
    df['ts_date'] = pd.to_datetime(df['ts_date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    if df['ts_date'].isna().all():
        df['ts_date'] = pd.to_datetime(df['ts_date'].str.split('.').str[0], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    df['ts_date'] = df['ts_date'].ffill()
    print(f"  ts_date: {df['ts_date'].iloc[0]} → {df['ts_date'].iloc[-1]}")

    missing = [col for col in target_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang: {missing}")

    df_sel = df[target_columns].copy()
    df_sel = df_sel.apply(pd.to_numeric, errors='coerce').ffill().bfill()

    # Potong ke max_rows (paling pendek)
    df_sel = df_sel.iloc[:max_rows]
    print(f"  Dipotong ke {len(df_sel)} baris")
    return df_sel

# =============================
# 4. Proses per Kategori: Potong ke Panjang Sama
# =============================
def process_category(files, category_name):
    print(f"\n=== PROSES {category_name.upper()} ===")
    dfs = []
    for file in files:
        df = pd.read_csv(file, encoding='utf-8-sig', sep=';')
        df.columns = [col.strip() for col in df.columns]
        ts_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
        df = df.rename(columns={ts_col: 'ts_date'})
        df[target_columns] = df[target_columns].apply(pd.to_numeric, errors='coerce')
        dfs.append(df[target_columns].ffill().bfill())

    # Cari panjang TERPENDEK di 3 file
    min_len = min(len(df) for df in dfs)
    print(f"{category_name} → panjang terpendek: {min_len:,} baris")

    # Potong semua ke panjang yang sama
    trimmed_dfs = [df.iloc[:min_len] for df in dfs]

    # Gabung 3 file jadi 1 hari representatif
    base_df = pd.concat(trimmed_dfs, ignore_index=True)
    base_df = base_df.ffill().bfill()
    print(f"{category_name} base (gabungan 3 file): {len(base_df):,} baris")
    return base_df, min_len

# Proses masing-masing kategori
healthy_files = csv_files_sorted[:3]
anomaly_files = csv_files_sorted[3:6]
failure_files = csv_files_sorted[6:]

df_healthy_base, healthy_rows_per_day = process_category(healthy_files, "Healthy")
df_anomaly_base, anomaly_rows_per_day = process_category(anomaly_files, "Anomaly")
df_failure_base, failure_rows_per_day = process_category(failure_files, "Failure")

# =============================
# 5. Duplikasi + Timestamp (Aman & Cocok)
# =============================
def duplicate_with_timestamp(base_df, n_days, start_time, rows_per_day):
    if len(base_df) == 0:
        return pd.DataFrame(), start_time

    # Ulangi data
    data_repeated = np.tile(base_df.values, (n_days, 1))
    df_out = pd.DataFrame(data_repeated, columns=target_columns)

    # Buat timestamp
    timestamps = []
    current = start_time
    total_needed = len(df_out)
    for _ in range(total_needed):
        timestamps.append(current)
        current += timedelta(minutes=1)

    df_out['ts_date'] = timestamps
    end_time = current
    return df_out, end_time

# Mulai dari 2024-01-01
start_time = datetime(2024, 1, 1, 0, 0)

# --- 30 hari sehat ---
df_healthy, start_time = duplicate_with_timestamp(df_healthy_base, 3, start_time, healthy_rows_per_day)

# --- 20 hari anomali ---
df_anomaly, start_time = duplicate_with_timestamp(df_anomaly_base, 2, start_time, anomaly_rows_per_day)

# --- 10 hari gagal ---
df_failure, _ = duplicate_with_timestamp(df_failure_base, 1, start_time, failure_rows_per_day)

# Gabung semua
df_combined = pd.concat([df_healthy, df_anomaly, df_failure], ignore_index=True)
print(f"\nFINAL: Total data: {len(df_combined):,} baris")
print(f"Rentang: {df_combined['ts_date'].min()} → {df_combined['ts_date'].max()}")

# =============================
# 6. Preprocessing & Normalisasi
# =============================
df_features = df_combined[target_columns].copy()
df_features = df_features.ffill().bfill()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_features)

# =============================
# 7. Buat Sequence
# =============================
def create_sequences(data, window_size=20000):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 20000
X, y = create_sequences(scaled_data, window_size)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# =============================
# 8. Train-Test Split
# =============================
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# =============================
# 9. Model LSTM
# =============================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2])),
    LSTM(64, return_sequences=False),
    Dense(50, activation='relu'),
    Dense(y.shape[1])
])
model.compile(optimizer='adam', loss='mse')
print(model.summary())

# =============================
# 10. Training
# =============================
print("\nMulai training...")
history = model.fit(
    X_train, y_train,
    epochs=20,           # dikurangi agar cepat
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# =============================
# 11. Prediksi & Visualisasi
# =============================
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

timestamps_test = df_combined['ts_date'].values[window_size:window_size + len(y_test)]

plot_cols = target_columns[:6]
plt.figure(figsize=(18, 12))
for i, col in enumerate(plot_cols):
    plt.subplot(3, 2, i + 1)
    plt.plot(timestamps_test, y_test_actual[:, i], label=f'Actual {col}')
    plt.plot(timestamps_test, y_pred[:, i], '--', label=f'Prediksi {col}', alpha=0.8)
    plt.title(col)
    plt.xlabel('Waktu')
    plt.ylabel('Nilai')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================
# 12. Simpan Model & Scaler
# =============================
model.save("lstm_forecast_siv.h5")
joblib.dump(scaler, "scaler_siv.pkl")
print("\nModel dan scaler disimpan:")
print("   -> lstm_forecast_siv.h5")
print("   -> scaler_siv.pkl")
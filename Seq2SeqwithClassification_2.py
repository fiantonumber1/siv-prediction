# =============================
# FULL CODE: MULTI-TASK SEQ2SEQ + PLUGGABLE LABELING FUNCTION
# - Fungsi labeling bisa diganti tanpa ubah kode utama
# - 0 = Sehat, 1 = Pre-Anomali, 2 = Near-Fail
# - RAM < 1GB | No NaN | 9 hari OK
# =============================
# A Multi-Task Sequence-to-Sequence Model for Time-Series Forecasting and Equipment Health Classification


import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
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

min_len = min(len(df) for df in all_dfs)
all_dfs = [df.iloc[:min_len] for df in all_dfs]
POINTS_PER_DAY = min_len
print(f"\nDipotong ke: {POINTS_PER_DAY:,} baris/hari")

# =============================
# 6. KOMPRESI 1:100 → 1500 titik/hari
# =============================
print(f"\nKOMPRESI 1:100 → {COMPRESSED_POINTS_PER_DAY:,} titik/hari")
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
# 7. FUNGSI LABELING (BISA DIGANTI DI SINI SAJA!)
# =============================
def label_health_status(df_day: pd.DataFrame) -> tuple:
    """
    LABELING FUNCTION — GANTI DI SINI SAJA!
    
    Input: DataFrame 1 hari (1500 baris, sudah dikompresi)
    Output: (status, reason)
        status: 0=Sehat, 1=Pre-Anomali, 2=Near-Fail
        reason: string penjelasan
    """
    energy = df_day['SIV_Output_Energy']
    max_energy = energy.max()
    if max_energy == 0:
        return 0, "No energy data"
    
    # Deteksi kegagalan: penurunan >50% dari max
    drop = energy.diff()
    failures = ((drop < -0.5 * max_energy) & (drop < 0)).sum()
    
    if failures == 0:
        return 0, "No failure"
    elif failures == 1:
        return 1, "1 failure"
    else:
        return 2, f"{failures} failures"

# =============================
# 8. TERAPKAN LABELING KE SEMUA HARI
# =============================
print("\nLABEL HEALTH STATUS:")
health_status = []
status_reasons = []

for i, df_day in enumerate(compressed_dfs):
    status, reason = label_health_status(df_day)
    health_status.append(status)
    status_reasons.append(reason)
    print(f"  Day {i+1}: {status} ({reason})")

# Simpan label
status_df = pd.DataFrame({
    'day': [f"Day {i+1}" for i in range(len(compressed_dfs))],
    'file': valid_files,
    'health_status': health_status,
    'reason': status_reasons
})
status_df.to_csv("health_status_per_day.csv", index=False)
print("DISIMPAN: health_status_per_day.csv")

# =============================
# 9. SIAPKAN DATA MULTI-TASK (3→1, Sliding Window)
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

# Normalisasi
scaler = MinMaxScaler()
X_flat = X_seq.reshape(-1, n_features)
X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape)
y_signal_flat = y_signal.reshape(-1, n_features)
y_signal_scaled = scaler.transform(y_signal_flat).reshape(y_signal.shape)

# =============================
# 10. MODEL: MULTI-TASK SEQ2SEQ + CLASSIFICATION
# =============================
input_seq = Input(shape=(WINDOW, n_features))
encoded = LSTM(128)(input_seq)
encoded = Dropout(0.2)(encoded)

# Decoder: Forecast
decoded = RepeatVector(FUTURE)(encoded)
decoded = LSTM(64, return_sequences=True)(decoded)
output_signal = TimeDistributed(Dense(n_features), name='signal')(decoded)

# Classifier: Status
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
print("\nTraining Multi-Task Model...")
history = model.fit(
    X_scaled, [y_signal_scaled, y_status],
    epochs=100, batch_size=1, verbose=1
)

# =============================
# 11. PREDIKSI HARI TERAKHIR
# =============================
last_input = X_scaled[-1:].reshape(1, WINDOW, n_features)
pred_signal_scaled, pred_status_prob = model.predict(last_input, verbose=0)
pred_signal = scaler.inverse_transform(pred_signal_scaled.reshape(-1, n_features)).reshape(FUTURE, n_features)
pred_status = np.argmax(pred_status_prob, axis=1)[0]
pred_confidence = np.max(pred_status_prob) * 100

status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
print(f"\nPREDIKSI HARI TERAKHIR: {status_map[pred_status]} ({pred_confidence:.1f}% confidence)")

# =============================
# 12. PLOT SEMUA PARAMETER + STATUS
# =============================
df_final = pd.concat(compressed_dfs, ignore_index=True)
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

# PERBAIKAN: Buat N+1 boundaries untuk N hari
n_days = len(compressed_dfs)
day_boundaries = np.arange(0, (n_days + 1) * COMPRESSED_POINTS_PER_DAY, COMPRESSED_POINTS_PER_DAY)

# Plot garis pemisah (kecuali ujung)
for pos in day_boundaries:
    if 0 < pos < len(df_final):
        ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9)

# PERBAIKAN: Hitung mid_points hanya sampai N hari
mid_points = [(day_boundaries[i] + day_boundaries[i+1]) // 2 for i in range(n_days)]

for i, mid in enumerate(mid_points):
    ax.text(mid, 1.05, f'Day {i+1}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='red',
            transform=ax.get_xaxis_transform())
    status_text = status_map[health_status[i]]
    color = ['green', 'orange', 'red'][health_status[i]]
    ax.text(mid, 1.15, status_text, ha='center', va='bottom',
            fontsize=10, fontweight='bold', color=color,
            transform=ax.get_xaxis_transform())

ax.set_xticks(mid_points)
ax.set_xticklabels([f'Day {i+1}' for i in range(n_days)])
ax.set_xlim(0, len(df_final))
ax.set_title(f"Semua Parameter + Health Status - {n_days} Hari", fontsize=14)
ax.set_xlabel("Hari", fontsize=12)
ax.set_ylabel("Nilai Normalisasi [0-1]")
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("plot_all_parameters_with_status.png", dpi=300, bbox_inches='tight')
print("Plot disimpan: plot_all_parameters_with_status.png")
plt.close()

# =============================
# 13. VISUALISASI: 3 PLOT (PAKAI SCALER UTAMA!)
# =============================
if len(compressed_dfs) < 4:
    print("ERROR: Data kurang dari 4 hari! Visualisasi 4 hari terakhir dibatalkan.")
else:
    all_dfs_4 = compressed_dfs[-4:]
    n_days_plot = len(all_dfs_4)
    df_plot = pd.concat(all_dfs_4, ignore_index=True)
    X_full = df_plot[target_columns].values.astype('float32')
    y_pred_day4 = pred_signal
    y_true_day4 = compressed_dfs[-1][target_columns].values.astype('float32')

    x_full = np.arange(n_days_plot * COMPRESSED_POINTS_PER_DAY)
    
    # PERBAIKAN: Gunakan scaler UTAMA (yang sama dengan training)
    X_flat = X_full.reshape(-1, n_features)
    X_norm = scaler.transform(X_flat).reshape(X_full.shape)  # PAKAI .transform(), bukan fit!

    y_pred_flat = y_pred_day4.reshape(-1, n_features)
    y_pred_norm = scaler.transform(y_pred_flat).reshape(y_pred_day4.shape)  # Sama!

    # day boundaries
    day_boundaries = np.arange(0, (n_days_plot + 1) * COMPRESSED_POINTS_PER_DAY, COMPRESSED_POINTS_PER_DAY)
    mid_points = [(day_boundaries[i] + day_boundaries[i+1]) // 2 for i in range(n_days_plot)]

    def setup_plot(ax, title):
        for pos in day_boundaries:
            if 0 < pos < len(x_full):
                ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
        for i, mid in enumerate(mid_points):
            day_idx = len(compressed_dfs) - n_days_plot + i
            ax.text(mid, 1.05, f'Day {day_idx+1}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='red',
                    transform=ax.get_xaxis_transform())
            status_text = status_map[health_status[day_idx]]
            color = ['green', 'orange', 'red'][health_status[day_idx]]
            ax.text(mid, 1.15, status_text, ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(mid_points)
        ax.set_xticklabels([f'Day {len(compressed_dfs)-n_days_plot+1+i}' for i in range(n_days_plot)])
        ax.set_xlim(0, len(x_full))
        ax.set_ylim(-0.05, 1.2)  # Batasi Y agar konsisten
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Hari (1.500 titik per hari)", fontsize=12)
        ax.set_ylabel("Nilai Normalisasi [0-1]")
        ax.grid(True, alpha=0.3)

    # --- GAMBAR 1 ---
    fig1, ax1 = plt.subplots(figsize=(20, 8))
    for i, col in enumerate(target_columns):
        ax1.plot(x_full, X_norm[:, i], label=col, linewidth=0.9, alpha=0.7)
    setup_plot(ax1, f'GAMBAR 1: 4 Hari Real + Health Status')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar1_4hari_real_with_status.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- GAMBAR 2 ---
    fig2, ax2 = plt.subplots(figsize=(20, 8))
    handles2 = []
    for i, col in enumerate(target_columns):
        line = ax2.plot(x_full[:3*COMPRESSED_POINTS_PER_DAY], X_norm[:3*COMPRESSED_POINTS_PER_DAY, i],
                        label=col, linewidth=0.9, alpha=0.7)[0]
        handles2.append(line)
    for i, col in enumerate(target_columns):
        ax2.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], y_pred_norm[:, i],
                 '--', linewidth=1.8, alpha=0.9)
    setup_plot(ax2, f'GAMBAR 2: 3 Hari Input + 1 Hari Prediksi + Status')
    ax2.legend(handles2, target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar2_input_plus_prediksi_with_status.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- GAMBAR 3 ---
    fig3, ax3 = plt.subplots(figsize=(20, 8))
    handles3 = []
    for i, col in enumerate(target_columns):
        line = ax3.plot(x_full[:3*COMPRESSED_POINTS_PER_DAY], X_norm[:3*COMPRESSED_POINTS_PER_DAY, i],
                        label=col, linewidth=0.9, alpha=0.7)[0]
        handles3.append(line)
    for i, col in enumerate(target_columns):
        ax3.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], X_norm[3*COMPRESSED_POINTS_PER_DAY:, i],
                 linewidth=1.2, alpha=0.8, label=f'Real {col}' if i == 0 else None)
        ax3.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], y_pred_norm[:, i],
                 '--', linewidth=1.8, alpha=0.9, label=f'Pred {col}' if i == 0 else None)
    setup_plot(ax3, f'GAMBAR 3: Day 4 → Real vs Prediksi + Status')
    ax3.legend(handles3, target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar3_real_vs_prediksi_with_status.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================
# 14. SIMPAN HASIL
# =============================
model.save("multitask_seq2seq_classification.h5")
joblib.dump(scaler, "scaler_multitask.pkl")

result_df = pd.DataFrame({
    'ts_date': compressed_dfs[-1]['ts_date'].values,
    'health_status_pred': [pred_status] * FUTURE,
    'confidence_%': [pred_confidence] * FUTURE
})
for i, col in enumerate(target_columns):
    result_df[f'actual_{col}'] = y_true_day4[:, i]
    result_df[f'pred_{col}'] = pred_signal[:, i]

result_df.to_csv("hasil_prediksi_dan_status.csv", index=False)

print("\nSELESAI! SEMUA FILE DISIMPAN:")
print("   -> health_status_per_day.csv")
print("   -> plot_all_parameters_with_status.png")
print("   -> gambar1_4hari_real_with_status.png")
print("   -> gambar2_input_plus_prediksi_with_status.png")
print("   -> gambar3_real_vs_prediksi_with_status.png")
print("   -> multitask_seq2seq_classification.h5")
print("   -> scaler_multitask.pkl")
print("   -> hasil_prediksi_dan_status.csv")
print(f"   -> Prediksi: {status_map[pred_status]} ({pred_confidence:.1f}%)")
# =============================
# FULL CODE PYTORCH + GPU + CHECKPOINTING (LANJUTKAN KALAU PUTUS)
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==================================================================
# TOMBOL UTAMA
# ==================================================================
USE_REAL_DATA_MODE = False
N_DUPLICATES = 100         # Bisa 1000, 5000, atau berapapun
N_EPOCHS = 1000              # Total epoch yang diinginkan
CHECKPOINT_INTERVAL = 50     # Simpan tiap berapa epoch
CHECKPOINT_DIR = "checkpoints"
# ==================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Buat folder checkpoint kalau belum ada
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

# =============================
# 1. BACA FILE CSV (sama seperti sebelumnya)
# =============================
# ... (kode baca CSV, crop, kompresi, duplikasi identik tetap 100% sama) ...
# (Saya skip bagian ini karena panjang & tidak berubah)

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files 
             if len(os.path.basename(f)) >= 12 
             and os.path.basename(f)[8:12] == ".csv"
             and "hasil" not in os.path.basename(f).lower()]

if len(csv_files) == 0:
    raise FileNotFoundError("Tidak ada CSV ditemukan!")

template_file = csv_files[0]
print(f"Template: {os.path.basename(template_file)} → duplikasi {N_DUPLICATES}x")

target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive'
]

START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600
N_TAKE = 150_000
COMPRESSION_FACTOR = 100
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR

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

# Kompresi & duplikasi identik
chunks, ts_mid = [], []
for i in range(COMPRESSED_POINTS_PER_DAY):
    s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
    chunks.append(template_raw[target_columns].iloc[s:e].mean())
    ts_mid.append(template_raw['ts_date'].iloc[s + COMPRESSION_FACTOR//2])

template_compressed = pd.DataFrame(chunks, columns=target_columns)
template_compressed.insert(0, 'ts_date', ts_mid)

compressed_dfs = []
valid_files = []
for day_idx in range(N_DUPLICATES):
    df_day = template_compressed.copy()
    offset = timedelta(days=day_idx)
    df_day['ts_date'] = df_day['ts_date'].dt.normalize() + offset + (df_day['ts_date'] - df_day['ts_date'].dt.normalize())
    compressed_dfs.append(df_day)
    valid_files.append(f"Identik_Day_{day_idx+1:02d}")

print(f"Berhasil buat {N_DUPLICATES} hari identik")

# Labeling (sama)
def label_health_status(df_day: pd.DataFrame) -> tuple:
    energy = df_day['SIV_Output_Energy']
    max_energy = energy.max()
    if max_energy == 0:
        return 0, "No energy data"
    drop = energy.diff()
    failures = ((drop < -0.5 * max_energy) & (drop < 0)).sum()
    if failures == 0:
        return 0, "No failure"
    elif failures == 1:
        return 1, "1 failure"
    else:
        return 2, f"{failures} failures"

print("\nLABEL HEALTH STATUS:")
health_status = [label_health_status(df)[0] for df in compressed_dfs]

# Siapkan data
WINDOW = 3 * COMPRESSED_POINTS_PER_DAY
FUTURE = COMPRESSED_POINTS_PER_DAY
n_features = len(target_columns)

X_seq, y_signal, y_status = [], [], []
for i in range(len(compressed_dfs) - 3):
    X_seq.append(np.concatenate([df[target_columns].values for df in compressed_dfs[i:i+3]], axis=0))
    y_signal.append(compressed_dfs[i+3][target_columns].values)
    y_status.append(health_status[i+3])

X_seq = np.array(X_seq, dtype=np.float32)
y_signal = np.array(y_signal, dtype=np.float32)
y_status = np.array(y_status, dtype=np.int64)

# Scaler
scaler = MinMaxScaler(feature_range=(-0.2, 1.2))
X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
y_signal_scaled = scaler.transform(y_signal.reshape(-1, n_features)).reshape(y_signal.shape)

X_tensor = torch.from_numpy(X_scaled).to(device)
y_signal_tensor = torch.from_numpy(y_signal_scaled).to(device)
y_status_tensor = torch.from_numpy(y_status).to(device)

# Dataset
class SeqDataset(Dataset):
    def __init__(self, X, y_sig, y_stat):
        self.X, self.y_sig, self.y_stat = X, y_sig, y_stat
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_sig[idx], self.y_stat[idx]

dataset = SeqDataset(X_tensor, y_signal_tensor, y_status_tensor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model
class MultiTaskSeq2Seq(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.LSTM(n_features, 128, batch_first=True)
        self.decoder = nn.LSTM(128, 64, batch_first=True)
        self.signal_out = nn.Linear(64, n_features)
        self.status_hidden = nn.Linear(128, 32)
        self.status_out = nn.Linear(32, 3)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        h_n = h_n.squeeze(0)
        dec_input = h_n.unsqueeze(1).repeat(1, FUTURE, 1)
        dec_out, _ = self.decoder(dec_input)
        signal_pred = self.signal_out(dec_out)
        status_h = torch.relu(self.status_hidden(h_n))
        status_pred = self.status_out(status_h)
        return signal_pred, status_pred

model = MultiTaskSeq2Seq(n_features).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_mse = nn.MSELoss()
criterion_ce = nn.CrossEntropyLoss()

# =============================
# CHECKPOINT: Cek apakah ada checkpoint
# =============================
start_epoch = 1
checkpoint_path = None

# Cari checkpoint terakhir
checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
if checkpoint_files:
    # Ambil yang epoch terbesar
    epochs = [int(f.split('_')[-1].replace('.pth', '')) for f in checkpoint_files]
    latest_epoch = max(epochs)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{latest_epoch}.pth")
    
    if latest_epoch < N_EPOCHS:
        print(f"Melanjutkan training dari epoch {latest_epoch + 1} (checkpoint ditemukan)")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = latest_epoch + 1
    else:
        print(f"Training sudah selesai di epoch {latest_epoch}. Tidak perlu lanjut.")
        start_epoch = N_EPOCHS + 1  # Skip training

else:
    print("Tidak ada checkpoint. Mulai training dari awal.")

# =============================
# TRAINING DENGAN CHECKPOINT
# =============================
if start_epoch <= N_EPOCHS:
    print(f"\nTraining dari epoch {start_epoch} sampai {N_EPOCHS}...")
    model.train()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        total_loss = 0.0
        for x_batch, y_sig_batch, y_stat_batch in dataloader:
            optimizer.zero_grad()
            sig_pred, stat_pred = model(x_batch)
            loss_sig = criterion_mse(sig_pred, y_sig_batch)
            loss_stat = criterion_ce(stat_pred, y_stat_batch)
            loss = loss_sig + 3.0 * loss_stat
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == N_EPOCHS:
            print(f"Epoch {epoch:4d}/{N_EPOCHS} | Loss: {total_loss:.6f}")

        # Simpan checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            cp_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': total_loss
            }, cp_path)
            print(f"   → Checkpoint disimpan: {cp_path}")

# =============================
# SELESAI TRAINING → PREDIKSI & PLOT (sama seperti sebelumnya)
# =============================
model.eval()
with torch.no_grad():
    last_input = X_tensor[-1:].to(device)
    pred_signal_scaled, pred_status_prob = model(last_input)
    pred_signal_scaled = pred_signal_scaled.cpu().numpy()[0]
    pred_status_prob = pred_status_prob.cpu().numpy()[0]

pred_signal = scaler.inverse_transform(pred_signal_scaled.reshape(-1, n_features)).reshape(FUTURE, n_features)
pred_status = np.argmax(pred_status_prob)
pred_confidence = np.max(pred_status_prob) * 100

status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
print(f"\nPREDIKSI HARI TERAKHIR: {status_map[pred_status]} ({pred_confidence:.1f}% confidence)")

# =============================
# SEMUA PLOT — 100% SAMA DENGAN VERSI TF
# =============================
df_final = pd.concat(compressed_dfs, ignore_index=True)
data_norm = df_final[target_columns].copy()
for col in target_columns:
    mn, mx = data_norm[col].min(), data_norm[col].max()
    data_norm[col] = 0 if mx - mn < 1e-8 else (data_norm[col] - mn) / (mx - mn)

x_index = np.arange(len(df_final))
fig, ax = plt.subplots(figsize=(20, 8))
for col in target_columns:
    ax.plot(x_index, data_norm[col], linewidth=0.9, alpha=0.7)

n_days = len(compressed_dfs)
day_boundaries = np.arange(0, (n_days + 1) * COMPRESSED_POINTS_PER_DAY, COMPRESSED_POINTS_PER_DAY)
mid_points = [(day_boundaries[i] + day_boundaries[i+1]) // 2 for i in range(n_days)]

for pos in day_boundaries[1:-1]:
    ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
for i, mid in enumerate(mid_points):
    ax.text(mid, 1.05, f'Day {i+1}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='red',
            transform=ax.get_xaxis_transform())
    color = ['green', 'orange', 'red'][health_status[i]]
    ax.text(mid, 1.15, status_map[health_status[i]], ha='center', va='bottom', fontsize=10, fontweight='bold', color=color,
            transform=ax.get_xaxis_transform())

ax.set_xlim(0, len(df_final))
ax.set_title(f"Semua Parameter + Health Status - {n_days} Hari (DATA IDENTIK)", fontsize=14)
ax.set_xlabel("Hari")
ax.set_ylabel("Nilai Normalisasi [0-1]")
ax.grid(True, alpha=0.3)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("plot_all_parameters_with_status.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================
# 4 HARI TERAKHIR + PREDIKSI (3 GAMBAR)
# =============================
if len(compressed_dfs) >= 4:
    all_dfs_4 = compressed_dfs[-4:]
    df_plot = pd.concat(all_dfs_4, ignore_index=True)
    X_full = df_plot[target_columns].values.astype('float32')
    y_true_day4 = compressed_dfs[-1][target_columns].values.astype('float32')

    x_full = np.arange(len(df_plot))
    X_flat = X_full.reshape(-1, n_features)
    X_norm = scaler.transform(X_flat).reshape(X_full.shape)

    y_pred_flat = pred_signal.reshape(-1, n_features)
    y_pred_norm = scaler.transform(y_pred_flat).reshape(pred_signal.shape)

    day_boundaries = np.arange(0, 5 * COMPRESSED_POINTS_PER_DAY, COMPRESSED_POINTS_PER_DAY)
    mid_points = [(day_boundaries[i] + day_boundaries[i+1]) // 2 for i in range(4)]

    def setup_plot(ax, title):
        for pos in day_boundaries[1:]:
            if pos < len(x_full):
                ax.axvline(x=pos, color='red', linestyle='--', linewidth=1.5, alpha=0.9)
        for i, mid in enumerate(mid_points):
            day_idx = len(compressed_dfs) - 4 + i
            ax.text(mid, 1.05, f'Day {day_idx+1}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='red',
                    transform=ax.get_xaxis_transform())
            status_text = status_map[health_status[day_idx]]
            color = ['green', 'orange', 'red'][health_status[day_idx]]
            ax.text(mid, 1.15, status_text, ha='center', va='bottom', fontsize=10, fontweight='bold', color=color,
                    transform=ax.get_xaxis_transform())
        ax.set_xlim(0, len(x_full))
        ax.set_ylim(-0.05, 1.2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Hari (1.500 titik per hari)", fontsize=12)
        ax.set_ylabel("Nilai Normalisasi [0-1]")
        ax.grid(True, alpha=0.3)

    # Gambar 1
    fig1, ax1 = plt.subplots(figsize=(20, 8))
    for i, col in enumerate(target_columns):
        ax1.plot(x_full, X_norm[:, i], label=col, linewidth=0.9, alpha=0.7)
    setup_plot(ax1, 'GAMBAR 1: 4 Hari Real + Health Status')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar1_4hari_real_with_status.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gambar 2
    fig2, ax2 = plt.subplots(figsize=(20, 8))
    handles = []
    for i, col in enumerate(target_columns):
        h = ax2.plot(x_full[:3*COMPRESSED_POINTS_PER_DAY], X_norm[:3*COMPRESSED_POINTS_PER_DAY, i],
                     label=col, linewidth=0.9, alpha=0.7)[0]
        handles.append(h)
    for i, col in enumerate(target_columns):
        ax2.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], y_pred_norm[:, i],
                 '--', linewidth=1.8, alpha=0.9)
    setup_plot(ax2, 'GAMBAR 2: 3 Hari Input + 1 Hari Prediksi + Status')
    ax2.legend(handles, target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar2_input_plus_prediksi_with_status.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gambar 3
    fig3, ax3 = plt.subplots(figsize=(20, 8))
    handles3 = []
    for i, col in enumerate(target_columns):
        h = ax3.plot(x_full[:3*COMPRESSED_POINTS_PER_DAY], X_norm[:3*COMPRESSED_POINTS_PER_DAY, i],
                     label=col, linewidth=0.9, alpha=0.7)[0]
        handles3.append(h)
    for i, col in enumerate(target_columns):
        ax3.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], X_norm[3*COMPRESSED_POINTS_PER_DAY:, i],
                 linewidth=1.2, alpha=0.8)
        ax3.plot(x_full[3*COMPRESSED_POINTS_PER_DAY:], y_pred_norm[:, i],
                 '--', linewidth=1.8, alpha=0.9)
    setup_plot(ax3, 'GAMBAR 3: Day 4 → Real vs Prediksi + Status')
    ax3.legend(handles3, target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar3_real_vs_prediksi_with_status.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================
# SIMPAN MODEL & HASIL
# =============================
torch.save(model.state_dict(), "multitask_seq2seq_classification.pth")
joblib.dump(scaler, "scaler_multitask.pkl")

result_df = pd.DataFrame({'ts_date': compressed_dfs[-1]['ts_date'].values})
for i, col in enumerate(target_columns):
    result_df[f'actual_{col}'] = y_true_day4[:, i]
    result_df[f'pred_{col}'] = pred_signal[:, i]
result_df['health_status_pred'] = pred_status
result_df['confidence_%'] = pred_confidence
result_df.to_csv("hasil_prediksi_dan_status.csv", index=False)

print("\nSELESAI 100%! Semua file sudah dibuat:")
print("   health_status_per_day.csv")
print("   plot_all_parameters_with_status.png")
print("   gambar1_4hari_real_with_status.png")
print("   gambar2_input_plus_prediksi_with_status.png")
print("   gambar3_real_vs_prediksi_with_status.png")
print("   multitask_seq2seq_classification.pth")
print("   scaler_multitask.pkl")
print("   hasil_prediksi_dan_status.csv")

# Hapus folder checkpoint kalau training sudah selesai (opsional)
if start_epoch > N_EPOCHS:
    import shutil
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"Folder {CHECKPOINT_DIR} dihapus (training selesai).")

print("\nSELESAI 100%! Semua file sudah dibuat + checkpoint aman.")
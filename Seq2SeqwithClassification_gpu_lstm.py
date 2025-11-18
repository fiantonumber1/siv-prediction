# =============================
# SEQ2SEQ + CLASSIFICATION 21 PARAMETER - FULL FINAL VERSION
# GPU + CHECKPOINT + RESUME + LOG + SEMUA PLOT + HASIL CSV
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
# TOMBOL UTAMA - SESUAIKAN SESUAI KEBUTUHAN
# ==================================================================
USE_REAL_DATA_MODE = True          # True = pakai file CSV asli | False = duplikasi identik
N_DUPLICATES = 20                   # hanya dipakai kalau False
N_EPOCHS = 600
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 50
CHECKPOINT_DIR = "checkpoints_21param_full"
LOG_FILE = "training_log_21param.txt"
# ==================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

# =============================
# 21 PARAMETER (FULL)
# =============================
target_columns = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive'
]
n_features = len(target_columns)

# =============================
# BACA & PREPROCESSING
# =============================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files if f.lower().endswith('.csv') and "hasil" not in os.path.basename(f).lower()]
csv_files_sorted = sorted(csv_files, key=extract_date)

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
    end_dt = datetime.combine(file_date, END_TIME)
    df = df[(df['ts_date'] >= start_dt) & (df['ts_date'] <= end_dt)]
    if len(df) < N_DROP_FIRST + N_TAKE * 0.8:
        return pd.DataFrame()
    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)
    return df[['ts_date'] + target_columns]

# Proses semua file
compressed_dfs = []
for f in csv_files_sorted:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"Skip {os.path.basename(f)} → data kurang")
        continue
    # Kompresi 100:1
    chunks, ts_mid = [], []
    for i in range(COMPRESSED_POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunks.append(df_raw[target_columns].iloc[s:e].mean())
        ts_mid.append(df_raw['ts_date'].iloc[s + COMPRESSION_FACTOR//2])
    df_comp = pd.DataFrame(chunks, columns=target_columns)
    df_comp.insert(0, 'ts_date', ts_mid)
    compressed_dfs.append(df_comp)

if not USE_REAL_DATA_MODE and len(compressed_dfs) >= 1:
    template = compressed_dfs[0]
    compressed_dfs = []
    for i in range(N_DUPLICATES):
        df_day = template.copy()
        offset = timedelta(days=i)
        df_day['ts_date'] = df_day['ts_date'].dt.normalize() + offset + (df_day['ts_date'] - df_day['ts_date'].dt.normalize())
        compressed_dfs.append(df_day)

print(f"\nTotal hari valid setelah preprocessing: {len(compressed_dfs)} hari")
if len(compressed_dfs) < 4:
    raise ValueError("Minimal 4 hari untuk training!")

# =============================
# LABELING HEALTH STATUS
# =============================
def label_health_status(df_day):
    e = df_day['SIV_Output_Energy']
    max_e = e.max()
    if max_e == 0: return 0, "No energy"
    drop = e.diff().dropna()
    fail = ((drop < -0.5 * max_e) & (drop < 0)).sum()
    if fail == 0: return 0, "Sehat"
    elif fail == 1: return 1, "Pre-Anomali"
    else: return 2, f"Near-Fail ({fail} drop)"

health_status = []
for i, df in enumerate(compressed_dfs):
    stat, txt = label_health_status(df)
    health_status.append(stat)
    print(f"Day {i+1:2d} → {txt}")

# =============================
# BUAT SEQUENCE 3 → 1
# =============================
WINDOW = 3 * COMPRESSED_POINTS_PER_DAY
FUTURE = COMPRESSED_POINTS_PER_DAY

X_seq, y_signal, y_status = [], [], []
for i in range(len(compressed_dfs) - 3):
    seq = np.concatenate([df[target_columns].values for df in compressed_dfs[i:i+3]], axis=0)
    X_seq.append(seq)
    y_signal.append(compressed_dfs[i+3][target_columns].values)
    y_status.append(health_status[i+3])

X_seq = np.array(X_seq, dtype=np.float32)
y_signal = np.array(y_signal, dtype=np.float32)
y_status = np.array(y_status, dtype=np.int64)

# Normalisasi
scaler = MinMaxScaler(feature_range=(-0.1, 1.1))
X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
y_scaled = scaler.transform(y_signal.reshape(-1, n_features)).reshape(y_signal.shape)

# Tensor
X_tensor = torch.FloatTensor(X_scaled).to(device)
y_sig_tensor = torch.FloatTensor(y_scaled).to(device)
y_stat_tensor = torch.LongTensor(y_status).to(device)

class SeqDataset(Dataset):
    def __init__(self, X, y_sig, y_stat):
        self.X, self.y_sig, self.y_stat = X, y_sig, y_stat
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_sig[idx], self.y_stat[idx]

dataloader = DataLoader(SeqDataset(X_tensor, y_sig_tensor, y_stat_tensor),
                        batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)  # ← UBAH DI SINI

# =============================
# MODEL MULTI-TASK
# =============================
class MultiTaskSeq2Seq(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.LSTM(n_features, 256, batch_first=True, dropout=0.3)
        self.decoder = nn.LSTM(256, 128, batch_first=True)
        self.sig_out = nn.Linear(128, n_features)
        self.cls_fc1 = nn.Linear(256, 64)
        self.cls_out = nn.Linear(64, 3)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.squeeze(0)

        dec_in = h.unsqueeze(1).repeat(1, FUTURE, 1)
        dec_out, _ = self.decoder(dec_in)
        sig_pred = self.sig_out(dec_out)

        cls_h = torch.relu(self.cls_fc1(h))
        stat_pred = self.cls_out(cls_h)
        return sig_pred, stat_pred

model = MultiTaskSeq2Seq(n_features).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_mse = nn.MSELoss()
criterion_ce = nn.CrossEntropyLoss()

# =============================
# CHECKPOINT & RESUME
# =============================
start_epoch = 1
latest_cp = None
checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
if checkpoint_files:
    epochs = [int(os.path.basename(f).split('_')[-1].replace('.pth', '')) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_cp = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{latest_epoch}.pth")
    if latest_epoch < N_EPOCHS:
        print(f"Melanjutkan training dari epoch {latest_epoch + 1}...")
        cp = torch.load(latest_cp, map_location=device)
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        start_epoch = latest_epoch + 1
    else:
        print(f"Training sudah selesai di epoch {latest_epoch}. Langsung ke prediksi.")
        start_epoch = N_EPOCHS + 1

# =============================
# TRAINING + LOG
# =============================
def log_print(text):
    print(text)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

log_print(f"\n=== TRAINING DIMULAI ===")
log_print(f"Total data training: {len(X_tensor)} sample")
log_print(f"Epoch target: {N_EPOCHS} | Batch size: {BATCH_SIZE}")

if start_epoch <= N_EPOCHS:
    model.train()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        total_loss = 0.0
        for x, y_sig, y_stat in dataloader:
            optimizer.zero_grad()
            sig_pred, stat_pred = model(x)
            loss_sig = criterion_mse(sig_pred, y_sig)
            loss_cls = criterion_ce(stat_pred, y_stat)
            loss = loss_sig + 3.0 * loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if epoch % 20 == 0 or epoch == N_EPOCHS:
            log_print(f"Epoch {epoch:4d}/{N_EPOCHS} | Loss: {avg_loss:.6f}")

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            cp_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, cp_path)
            log_print(f"   → Checkpoint disimpan: {cp_path}")

log_print("=== TRAINING SELESAI ===\n")

# =============================
# PREDIKSI HARI DEPAN
# =============================
model.eval()
with torch.no_grad():
    last_input = X_tensor[-1:].to(device)
    pred_sig_scaled, pred_stat_prob = model(last_input)
    pred_sig_scaled = pred_sig_scaled.cpu().numpy()[0]
    pred_stat_prob = pred_stat_prob.cpu().numpy()[0]

pred_signal = scaler.inverse_transform(pred_sig_scaled.reshape(-1, n_features)).reshape(FUTURE, n_features)
pred_status = np.argmax(pred_stat_prob)
pred_confidence = np.max(pred_stat_prob) * 100

status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
log_print(f"PREDIKSI HARI DEPAN: {status_map[pred_status]} ({pred_confidence:.2f}% confidence)")

# =============================
# SIMPAN MODEL & SCALER
# =============================
torch.save(model.state_dict(), "model_21param_final.pth")
joblib.dump(scaler, "scaler_21param.pkl")
log_print("Model & scaler disimpan")

# =============================
# PLOT 1: SEMUA HARI + STATUS
# =============================
df_all = pd.concat(compressed_dfs, ignore_index=True)
norm_all = df_all[target_columns].copy()
for col in target_columns:
    mn, mx = norm_all[col].min(), norm_all[col].max()
    if mx - mn > 1e-8:
        norm_all[col] = (norm_all[col] - mn) / (mx - mn)
    else:
        norm_all[col] = 0

x = np.arange(len(df_all))
fig, ax = plt.subplots(figsize=(24, 10))
for col in target_columns:
    ax.plot(x, norm_all[col], linewidth=0.9, alpha=0.7)
day_bounds = np.arange(0, (len(compressed_dfs)+1)*COMPRESSED_POINTS_PER_DAY, COMPRESSED_POINTS_PER_DAY)
for b in day_bounds[1:-1]:
    ax.axvline(b, color='red', linestyle='--', alpha=0.8)
mid_points = [(day_bounds[i] + day_bounds[i+1])//2 for i in range(len(compressed_dfs))]
for i, mid in enumerate(mid_points):
    ax.text(mid, 1.05, f'Day {i+1}', ha='center', color='red', fontweight='bold', transform=ax.get_xaxis_transform())
    ax.text(mid, 1.15, status_map[health_status[i]], ha='center', 
            color=['green','orange','red'][health_status[i]], fontweight='bold', transform=ax.get_xaxis_transform())
ax.set_title(f"21 Parameter + Health Status - {len(compressed_dfs)} Hari", fontsize=16)
ax.set_ylabel("Normalized [0-1]")
ax.grid(alpha=0.3)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("plot_all_parameters_with_status.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================
# PLOT 4 HARI TERAKHIR (3 GAMBAR)
# =============================
if len(compressed_dfs) >= 4:
    last4_dfs = compressed_dfs[-4:]
    df4 = pd.concat(last4_dfs, ignore_index=True)
    real4 = df4[target_columns].values
    norm4 = scaler.transform(real4.reshape(-1, n_features)).reshape(real4.shape)
    pred_norm = scaler.transform(pred_signal.reshape(-1, n_features)).reshape(pred_signal.shape)
    x4 = np.arange(len(df4))

    def setup_plot(ax, title):
        bounds = np.arange(0, 5*COMPRESSED_POINTS_PER_DAY, COMPRESSED_POINTS_PER_DAY)
        for b in bounds[1:]:
            if b < len(x4):
                ax.axvline(b, color='red', linestyle='--', alpha=0.8)
        mids = [(bounds[i] + bounds[i+1])//2 for i in range(4)]
        for i, m in enumerate(mids):
            day_idx = len(compressed_dfs) - 4 + i
            ax.text(m, 1.05, f'Day {day_idx+1}', ha='center', color='red', fontweight='bold', transform=ax.get_xaxis_transform())
            ax.text(m, 1.15, status_map[health_status[day_idx]], ha='center',
                    color=['green','orange','red'][health_status[day_idx]], fontweight='bold', transform=ax.get_xaxis_transform())
        ax.set_title(title, fontsize=15)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.1, 1.3)

    # Gambar 1
    fig, ax = plt.subplots(figsize=(24, 10))
    for i, col in enumerate(target_columns):
        ax.plot(x4, norm4[:, i], linewidth=1, alpha=0.8)
    setup_plot(ax, "GAMBAR 1: 4 Hari Real Data + Health Status")
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar1_4hari_real.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gambar 2
    fig, ax = plt.subplots(figsize=(24, 10))
    for i, col in enumerate(target_columns):
        ax.plot(x4[:3*COMPRESSED_POINTS_PER_DAY], norm4[:3*COMPRESSED_POINTS_PER_DAY, i], linewidth=1)
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], pred_norm[:, i], '--', linewidth=2)
    setup_plot(ax, "GAMBAR 2: 3 Hari Input + 1 Hari Prediksi")
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar2_input_plus_prediksi.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gambar 3
    fig, ax = plt.subplots(figsize=(24, 10))
    for i, col in enumerate(target_columns):
        ax.plot(x4[:3*COMPRESSED_POINTS_PER_DAY], norm4[:3*COMPRESSED_POINTS_PER_DAY, i], linewidth=1)
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], norm4[3*COMPRESSED_POINTS_PER_DAY:, i], linewidth=1.5)
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], pred_norm[:, i], '--', linewidth=2)
    setup_plot(ax, "GAMBAR 3: Real vs Prediksi (Hari Terakhir)")
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar3_real_vs_prediksi.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================
# SIMPAN HASIL PREDIKSI CSV
# =============================
result_df = pd.DataFrame(pred_signal, columns=target_columns)
result_df.insert(0, 'ts_date', compressed_dfs[-1]['ts_date'].values)
result_df['health_status_pred'] = status_map[pred_status]
result_df['confidence_percent'] = pred_confidence
result_df.to_csv("prediksi_hari_depan_21param.csv", index=False)
log_print("prediksi_hari_depan_21param.csv disimpan")

# =============================
# SELESAI
# =============================
log_print("\nSEMUA SELESAI 100%!")
log_print("File yang dihasilkan:")
log_print("   → model_21param_final.pth")
log_print("   → scaler_21param.pkl")
log_print("   → prediksi_hari_depan_21param.csv")
log_print("   → plot_all_parameters_with_status.png")
log_print("   → gambar1_4hari_real.png")
log_print("   → gambar2_input_plus_prediksi.png")
log_print("   → gambar3_real_vs_prediksi.png")
log_print("   → training_log_21param.txt")
log_print("   → semua checkpoint di folder: " + CHECKPOINT_DIR)

print("\n" + "="*60)
print("SEMUA SELESAI! COBA CEK FOLDER SEKARANG!")
print("="*60)
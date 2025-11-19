# =============================
# SEQ2SEQ + CLASSIFICATION 21 PARAMETER - FULL TCN VERSION
# VERSI FINAL KHUSUS DATA REAL 100% (TIDAK ADA DUPLIKASI SAMA SEKALI)
# + MSE FORECAST & ACCURACY KLASIFIKASI PER EPOCH (DISIMPAN DI LOG)
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
# TOMBOL UTAMA - UBAH DI SINI AJA SELAMANYA
# ==================================================================
N_EPOCHS = 1000
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 50
CHECKPOINT_DIR = "checkpoints_21param_TCN"
LOG_FILE = "training_log_21param_TCN.txt"

COMPRESSION_FACTOR = 10             # SATU-SATUNYA ANGKA YANG PERNAH KAMU UBAH
# 100 = cepat | 50 = sedang | 25 = detail | 10 = super detail
# ==================================================================

# ================== AUTO-CALCULATED — JANGAN DIUBAH MANUAL ==================
N_TAKE = 150_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR

WINDOW = 3 * COMPRESSED_POINTS_PER_DAY      # 3 hari input
FUTURE = COMPRESSED_POINTS_PER_DAY          # 1 hari prediksi

START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

print(f"\n[CONFIG] COMPRESSION_FACTOR = {COMPRESSION_FACTOR}×")
print(f"         → {COMPRESSED_POINTS_PER_DAY} titik = 1 hari penuh")
print(f"         → Input sequence  = {WINDOW} timesteps (3 hari)")
print(f"         → Prediksi output = {FUTURE} timesteps (1 hari)\n")

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
# BACA & PREPROCESSING (HANYA DATA REAL)
# =============================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files if f.lower().endswith('.csv') and "hasil" not in os.path.basename(f).lower()]
csv_files_sorted = sorted(csv_files, key=extract_date)

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

# Proses semua file (100% real, tidak ada duplikasi)
compressed_dfs = []
for f in csv_files_sorted:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"Skip {os.path.basename(f)} → data kurang")
        continue
    chunks, ts_mid = [], []
    for i in range(COMPRESSED_POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        chunks.append(df_raw[target_columns].iloc[s:e].mean())
        ts_mid.append(df_raw['ts_date'].iloc[s + COMPRESSION_FACTOR//2])
    df_comp = pd.DataFrame(chunks, columns=target_columns)
    df_comp.insert(0, 'ts_date', ts_mid)
    compressed_dfs.append(df_comp)

print(f"\nTotal hari valid (DATA REAL ASLI): {len(compressed_dfs)} hari")
if len(compressed_dfs) < 4:
    raise ValueError("Minimal 4 hari data real untuk training!")

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
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# =============================
# TCN MODEL (MULTI-TASK)
# =============================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
    def forward(self, x):
        return self.conv(x)[:, :, :-self.padding]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.dropout1(self.relu1(self.norm1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.norm2(self.conv2(out))))
        return out + residual

class TCNMultiTask(nn.Module):
    def __init__(self, n_features, n_channels=96, kernel_size=3, n_blocks=7, dropout=0.3):
        super().__init__()
        layers = []
        dilations = [1, 2, 4, 8, 16, 32, 64]
        in_ch = n_features
        for i in range(n_blocks):
            layers.append(ResidualBlock(in_ch, n_channels, kernel_size, dilations[i], dropout))
            in_ch = n_channels
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Linear(n_channels, n_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(n_channels, n_features * FUTURE)
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        feature = self.global_pool(out).squeeze(-1)
        sig_pred = self.decoder(feature).view(-1, FUTURE, n_features)
        stat_pred = self.classifier(feature)
        return sig_pred, stat_pred

model = TCNMultiTask(n_features=n_features).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)

criterion_mse = nn.MSELoss()
criterion_ce = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.5, 4.0]).to(device))

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
        scheduler.load_state_dict(cp['scheduler'])
        start_epoch = latest_epoch + 1
    else:
        print(f"Training SUDAH SELESAI di epoch {latest_epoch}. Langsung prediksi!")

# =============================
# LOG FUNCTION
# =============================
def log_print(text):
    print(text)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

log_print(f"\n{'='*80}")
log_print(f"TCN TRAINING DATA REAL 100% - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Total sample: {len(X_tensor)} | Epoch: {N_EPOCHS} | Batch: {BATCH_SIZE}")
log_print(f"COMPRESSION_FACTOR: {COMPRESSION_FACTOR}× → {COMPRESSED_POINTS_PER_DAY} pts/hari")
log_print(f"{'='*80}")

# =============================
# TRAINING LOOP + MSE & ACCURACY PER EPOCH
# =============================
if start_epoch <= N_EPOCHS:
    model.train()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        total_loss = total_mse = 0.0
        total_correct = total_samples = 0

        for x, y_sig, y_stat in dataloader:
            optimizer.zero_grad()
            sig_pred, stat_pred = model(x)
            loss_sig = criterion_mse(sig_pred, y_sig)
            loss_cls = criterion_ce(stat_pred, y_stat)
            loss = loss_sig + 3.5 * loss_cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mse += loss_sig.item()
            _, pred = torch.max(stat_pred, 1)
            total_correct += (pred == y_stat).sum().item()
            total_samples += y_stat.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        acc = 100.0 * total_correct / total_samples
        scheduler.step(avg_loss)

        log_line = f"Epoch {epoch:4d}/{N_EPOCHS} | Loss: {avg_loss:.6f} | MSE(Forecast): {avg_mse:.7f} | Acc(Class): {acc:6.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}"
        print(log_line)
        log_print(log_line)

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            cp_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'loss': avg_loss}, cp_path)
            log_print(f"   → Checkpoint: {cp_path}")

    log_print("=== TRAINING SELESAI ===\n")
else:
    model.load_state_dict(torch.load(latest_cp, map_location=device)['model'])

# =============================
# PREDIKSI & SIMPAN HASIL (sama seperti sebelumnya)
# =============================
model.eval()
with torch.no_grad():
    pred_sig_scaled, pred_stat_logits = model(X_tensor[-1:].to(device))
    pred_sig_scaled = pred_sig_scaled.cpu().numpy()[0]
    pred_stat_prob = torch.softmax(pred_stat_logits, dim=1).cpu().numpy()[0]
    pred_status = int(np.argmax(pred_stat_prob))
    pred_confidence = float(pred_stat_prob[pred_status] * 100)

pred_signal = scaler.inverse_transform(pred_sig_scaled.reshape(-1, n_features)).reshape(FUTURE, n_features)
status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
log_print(f"PREDIKSI HARI DEPAN: {status_map[pred_status]} ({pred_confidence:.2f}% confidence)")

torch.save(model.state_dict(), "model_21param_TCN_final.pth")
joblib.dump(scaler, "scaler_21param_TCN.pkl")
log_print("Model & scaler disimpan")

# =============================
# PLOT-PLOT (100% AKURAT BERAPAPUN KOMPRESI)
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

ax.set_title(f"21 Parameter + Health Status (TCN) - {len(compressed_dfs)} Hari", fontsize=16)
ax.set_ylabel("Normalized [0-1]")
ax.grid(alpha=0.3)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.savefig("plot_all_parameters_TCN.png", dpi=300, bbox_inches='tight')
plt.close()

# 4 Hari Terakhir + Prediksi
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
    setup_plot(ax, "GAMBAR 1: 4 Hari Real Data + Health Status (TCN)")
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig("gambar1_4hari_real_TCN.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gambar 2
    fig, ax = plt.subplots(figsize=(24, 10))
    for i, col in enumerate(target_columns):
        ax.plot(x4[:3*COMPRESSED_POINTS_PER_DAY], norm4[:3*COMPRESSED_POINTS_PER_DAY, i], linewidth=1.2, label=col)
    for i, col in enumerate(target_columns):
        color = ax.get_lines()[i].get_color()
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], pred_norm[:, i], '--', linewidth=2.8, color=color, alpha=0.95)
    setup_plot(ax, "GAMBAR 2: 3 Hari Input + 1 Hari Prediksi (TCN)")
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig("gambar2_input_plus_prediksi_TCN.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Gambar 3
    fig, ax = plt.subplots(figsize=(24, 10))
    for i, col in enumerate(target_columns):
        ax.plot(x4[:3*COMPRESSED_POINTS_PER_DAY], norm4[:3*COMPRESSED_POINTS_PER_DAY, i], linewidth=1.2)
    for i, col in enumerate(target_columns):
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], norm4[3*COMPRESSED_POINTS_PER_DAY:, i], linewidth=1.8, alpha=0.9)
    for i, col in enumerate(target_columns):
        color = ax.get_lines()[i].get_color()
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], pred_norm[:, i], '--', linewidth=3, color=color, alpha=0.95,
                label=f'Pred {col}' if i==0 else None)
    setup_plot(ax, "GAMBAR 3: Real vs Prediksi Hari Terakhir (TCN)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig("gambar3_real_vs_prediksi_TCN.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================
# SIMPAN HASIL PREDIKSI CSV
# =============================
result_df = pd.DataFrame(pred_signal, columns=target_columns)
result_df.insert(0, 'ts_date', compressed_dfs[-1]['ts_date'].values[-FUTURE:])
result_df['health_status_pred'] = status_map[pred_status]
result_df['confidence_percent'] = pred_confidence
result_df.to_csv("prediksi_hari_depan_21param_TCN.csv", index=False)
log_print("prediksi_hari_depan_21param_TCN.csv disimpan")

log_print("\nSEMUA SELESAI 100% - VERSI DATA REAL TANPA DUPLIKASI APAPUN!")
print("\n" + "="*90)
print("SELESAI TOTAL! Hanya pakai data asli, tidak ada duplikasi lagi.")
print("Ganti COMPRESSION_FACTOR = 10 / 25 / 50 / 100 → tetap akurat otomatis")
print("MSE + Accuracy tiap epoch sudah tercatat di training_log_21param_TCN.txt")
print("="*90)
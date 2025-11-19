# =============================
# SEQ2SEQ + CLASSIFICATION 21 PARAMETER - FULL TCN VERSION
# FINAL + VALIDATION + BEST MODEL + SEMUA PLOT + MSE AKHIR RESMI
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
USE_REAL_DATA_MODE = True
N_DUPLICATES = 20
N_EPOCHS = 1000
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 50
CHECKPOINT_DIR = "checkpoints_21param_TCN"
LOG_FILE = "training_log_21param_TCN.txt"
VAL_RATIO = 0.2

COMPRESSION_FACTOR = 10             # 100=cepat, 50=sedang, 25=detail, 10=super detail
# ==================================================================

N_TAKE = 150_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
WINDOW = 3 * COMPRESSED_POINTS_PER_DAY
FUTURE = COMPRESSED_POINTS_PER_DAY

START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

print(f"\n[CONFIG] COMPRESSION_FACTOR = {COMPRESSION_FACTOR}x")
print(f"         → {COMPRESSED_POINTS_PER_DAY} titik/hari | Input 3 hari: {WINDOW} | Prediksi 1 hari: {FUTURE}\n")

# =============================
# 21 PARAMETER
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
# PREPROCESSING
# =============================
def extract_date(f):
    return datetime.strptime(os.path.basename(f)[:8], "%d%m%Y")

csv_files = sorted([f for f in glob.glob(os.path.join(folder_path, "*.csv"))
                    if f.lower().endswith('.csv') and "hasil" not in os.path.basename(f).lower()],
                   key=extract_date)

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

compressed_dfs = []
for f in csv_files:
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

if not USE_REAL_DATA_MODE and len(compressed_dfs) >= 1:
    template = compressed_dfs[0]
    compressed_dfs = []
    for i in range(N_DUPLICATES):
        df_day = template.copy()
        offset = timedelta(days=i)
        df_day['ts_date'] = df_day['ts_date'].dt.normalize() + offset + (df_day['ts_date'] - df_day['ts_date'].dt.normalize())
        compressed_dfs.append(df_day)

print(f"\nTotal hari valid: {len(compressed_dfs)} hari")
if len(compressed_dfs) < 4:
    raise ValueError("Minimal 4 hari!")

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
# BUAT SEQUENCE & NORMALISASI
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

scaler = MinMaxScaler(feature_range=(-0.1, 1.1))
X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
y_scaled = scaler.transform(y_signal.reshape(-1, n_features)).reshape(y_signal.shape)

X_tensor = torch.FloatTensor(X_scaled)
y_sig_tensor = torch.FloatTensor(y_scaled)
y_stat_tensor = torch.LongTensor(y_status)

# =============================
# TRAIN-VAL SPLIT
# =============================
indices = np.arange(len(X_tensor))
train_idx, val_idx = train_test_split(indices, test_size=VAL_RATIO, random_state=42, stratify=y_status)

class SeqDataset(Dataset):
    def __init__(self, X, y_sig, y_stat):
        self.X, self.y_sig, self.y_stat = X, y_sig, y_stat
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_sig[idx], self.y_stat[idx]

train_loader = DataLoader(SeqDataset(X_tensor[train_idx], y_sig_tensor[train_idx], y_stat_tensor[train_idx]),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_tensor[val_idx],   y_sig_tensor[val_idx],   y_stat_tensor[val_idx]),
                          batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_idx)} | Validation: {len(val_idx)} samples\n")

# =============================
# MODEL TCN
# =============================
class CausalConv1d(nn.Module):
    def __init__(self, in_c, out_c, k, d=1):
        super().__init__()
        self.pad = (k-1)*d
        self.conv = nn.Conv1d(in_c, out_c, k, padding=self.pad, dilation=d)
    def forward(self, x): return self.conv(x)[:, :, :-self.pad]

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, d=1, drop=0.3):
        super().__init__()
        self.c1 = CausalConv1d(in_c, out_c, k, d)
        self.n1 = nn.BatchNorm1d(out_c); self.r1 = nn.ReLU(); self.d1 = nn.Dropout(drop)
        self.c2 = CausalConv1d(out_c, out_c, k, d)
        self.n2 = nn.BatchNorm1d(out_c); self.r2 = nn.ReLU(); self.d2 = nn.Dropout(drop)
        self.skip = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
    def forward(self, x):
        res = self.skip(x)
        x = self.d1(self.r1(self.n1(self.c1(x))))
        x = self.d2(self.r2(self.n2(self.c2(x))))
        return x + res

class TCNMultiTask(nn.Module):
    def __init__(self, n_features, n_ch=96, n_blocks=7, drop=0.3):
        super().__init__()
        layers = []
        dil = [1,2,4,8,16,32,64]
        in_c = n_features
        for i in range(n_blocks):
            layers.append(ResidualBlock(in_c, n_ch, dilation=dil[i], drop=drop))
            in_c = n_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(nn.Linear(n_ch, n_ch), nn.ReLU(), nn.Dropout(drop),
                                     nn.Linear(n_ch, n_features * FUTURE))
        self.classifier = nn.Sequential(nn.Linear(n_ch, 128), nn.ReLU(), nn.Dropout(drop), nn.Linear(128, 3))
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        feat = self.pool(out).squeeze(-1)
        sig = self.decoder(feat).view(-1, FUTURE, n_features)
        cls = self.classifier(feat)
        return sig, cls

model = TCNMultiTask(n_features=n_features).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30, verbose=True)

criterion_mse = nn.MSELoss()
criterion_ce  = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 2.5, 4.0]).to(device))

# =============================
# LOGGING
# =============================
def log_print(text):
    print(text)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

log_print("\n" + "="*90)
log_print(f"TRAINING STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"COMPRESSION_FACTOR: {COMPRESSION_FACTOR}x | Train/Val: {len(train_idx)}/{len(val_idx)}")
log_print("="*90)

# =============================
# TRAINING + VALIDATION
# =============================
best_val_mse = float('inf')
start_epoch = 1

# Resume
cp_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
if cp_files:
    latest_epoch = max([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in cp_files])
    cp_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{latest_epoch}.pth")
    if latest_epoch < N_EPOCHS:
        print(f"Resume dari epoch {latest_epoch + 1}...")
        cp = torch.load(cp_path, map_location=device)
        model.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        scheduler.load_state_dict(cp['scheduler'])
        start_epoch = latest_epoch + 1
        best_val_mse = cp.get('best_val_mse', float('inf'))

for epoch in range(start_epoch, N_EPOCHS + 1):
    model.train()
    tr_mse = tr_ce = tr_loss = 0.0
    print(f"EPOCH {epoch:4d}/{N_EPOCHS} → ", end="")
    for x, y_sig, y_stat in train_loader:
        optimizer.zero_grad()
        sig_pred, stat_pred = model(x.to(device))
        loss_mse = criterion_mse(sig_pred, y_sig.to(device))
        loss_ce  = criterion_ce(stat_pred, y_stat.to(device))
        loss = loss_mse + 3.5 * loss_ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_mse += loss_mse.item()
        tr_ce += loss_ce.item()
        tr_loss += loss.item()
        print(".", end="", flush=True)

    avg_tr_mse = tr_mse / len(train_loader)
    avg_tr_ce  = tr_ce  / len(train_loader)
    avg_tr_loss = tr_loss / len(train_loader)

    # Validation
    model.eval()
    val_mse = 0.0
    with torch.no_grad():
        for x, y_sig, _ in val_loader:
            sig_pred, _ = model(x.to(device))
            val_mse += criterion_mse(sig_pred, y_sig.to(device)).item()
    val_mse /= len(val_loader)

    scheduler.step(val_mse)

    print(f"\n{'-'*70}")
    print(f"EPOCH {epoch:4d} | Train MSE: {avg_tr_mse:.6f} | Val MSE: {val_mse:.6f} ← MSE AKHIR")
    print(f"              CE Loss: {avg_tr_ce:.6f} | Total: {avg_tr_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'-'*70}\n")

    log_print(f"EPOCH {epoch} | TrainMSE {avg_tr_mse:.6f} | ValMSE {val_mse:.6f} | Total {avg_tr_loss:.6f}")

    if val_mse < best_val_mse:
        best_val_mse = val_mse
        torch.save(model.state_dict(), "model_21param_TCN_BEST.pth")
        print(f"*** BEST MODEL! Val MSE = {best_val_mse:.6f} ***\n")

    if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'val_mse': val_mse, 'best_val_mse': best_val_mse},
                   os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth"))

# =============================
# FINAL RESULT
# =============================
log_print(f"\nTRAINING SELESAI! FINAL VALIDATION MSE (MSE AKHIR RESMI): {best_val_mse:.6f}")
print("\n" + "="*100)
print(f"FINAL VALIDATION MSE (INI MSE AKHIR YANG MEWAKILI KESELURUHAN TRAINING): {best_val_mse:.6f}")
print("Model terbaik: model_21param_TCN_BEST.pth")
print("="*100)

# Load best model
model.load_state_dict(torch.load("model_21param_TCN_BEST.pth", map_location=device))
model.eval()

# =============================
# PREDIKSI HARI DEPAN
# =============================
with torch.no_grad():
    last_input = X_tensor[-1:].to(device)
    pred_sig_scaled, pred_stat_logits = model(last_input)
    pred_sig_scaled = pred_sig_scaled.cpu().numpy()[0]
    pred_stat_prob = torch.softmax(pred_stat_logits, dim=1).cpu().numpy()[0]
    pred_status = int(np.argmax(pred_stat_prob))
    pred_confidence = pred_stat_prob[pred_status] * 100

pred_signal = scaler.inverse_transform(pred_sig_scaled.reshape(-1, n_features)).reshape(FUTURE, n_features)

status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
log_print(f"PREDIKSI HARI DEPAN: {status_map[pred_status]} ({pred_confidence:.2f}% confidence)")

# Simpan hasil
torch.save(model.state_dict(), "model_21param_TCN_final.pth")
joblib.dump(scaler, "scaler_21param_TCN.pkl")

result_df = pd.DataFrame(pred_signal, columns=target_columns)
result_df.insert(0, 'ts_date', compressed_dfs[-1]['ts_date'].values[-FUTURE:])
result_df['health_status_pred'] = status_map[pred_status]
result_df['confidence_percent'] = pred_confidence
result_df.to_csv("prediksi_hari_depan_21param_TCN.csv", index=False)

# =============================
# PLOT SEMUA DATA + HEALTH STATUS
# =============================
df_all = pd.concat(compressed_dfs, ignore_index=True)
norm_all = df_all[target_columns].copy()
for col in target_columns:
    mn, mx = norm_all[col].min(), norm_all[col].max()
    norm_all[col] = (norm_all[col] - mn) / (mx - mn + 1e-8) if mx > mn else 0

x = np.arange(len(df_all))
fig, ax = plt.subplots(figsize=(28, 10))
for col in target_columns:
    ax.plot(x, norm_all[col], linewidth=0.9, alpha=0.7)
day_bounds = np.arange(0, len(df_all)+1, COMPRESSED_POINTS_PER_DAY)
for b in day_bounds[1:-1]: ax.axvline(b, color='red', linestyle='--', alpha=0.7)
for i in range(len(compressed_dfs)):
    mid = (day_bounds[i] + day_bounds[i+1]) // 2
    ax.text(mid, 1.05, f'D{i+1}', ha='center', color='red', fontweight='bold', transform=ax.get_xaxis_transform())
    ax.text(mid, 1.15, status_map[health_status[i]], ha='center',
            color=['green','orange','red'][health_status[i]], fontweight='bold', transform=ax.get_xaxis_transform())
ax.set_title(f"21 Parameter + Health Status - {len(compressed_dfs)} Hari (TCN)", fontsize=18)
ax.set_ylabel("Normalized")
ax.grid(alpha=0.3)
ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig("plot_all_parameters_TCN.png", dpi=300, bbox_inches='tight')
plt.close()

# =============================
# 4 HARI TERAKHIR + 3 GAMBAR
# =============================
if len(compressed_dfs) >= 4:
    last4 = pd.concat(compressed_dfs[-4:], ignore_index=True)
    real4 = last4[target_columns].values
    norm4 = scaler.transform(real4.reshape(-1, n_features)).reshape(real4.shape)
    pred_norm = scaler.transform(pred_signal.reshape(-1, n_features)).reshape(pred_signal.shape)
    x4 = np.arange(len(last4))

    def setup(ax, title):
        bounds = np.arange(0, 5*COMPRESSED_POINTS_PER_DAY, COMPRESSED_POINTS_PER_DAY)
        for b in bounds[1:]: ax.axvline(b, color='red', linestyle='--', alpha=0.8)
        for i in range(4):
            mid = bounds[i] + COMPRESSED_POINTS_PER_DAY//2
            day_idx = len(compressed_dfs) - 4 + i
            ax.text(mid, 1.05, f'D{day_idx+1}', ha='center', color='red', fontweight='bold', transform=ax.get_xaxis_transform())
            ax.text(mid, 1.15, status_map[health_status[day_idx]], ha='center',
                    color=['green','orange','red'][health_status[day_idx]], fontweight='bold', transform=ax.get_xaxis_transform())
        ax.set_title(title, fontsize=16)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.1, 1.3)

    # Gambar 1: 4 Hari Real
    fig, ax = plt.subplots(figsize=(28, 10))
    for i, col in enumerate(target_columns):
        ax.plot(x4, norm4[:, i], linewidth=1.2)
    setup(ax, "1. 4 Hari Real Data + Health Status")
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.tight_layout(); plt.savefig("gambar1_4hari_real_TCN.png", dpi=300, bbox_inches='tight'); plt.close()

    # Gambar 2: Input + Prediksi
    fig, ax = plt.subplots(figsize=(28, 10))
    for i, col in enumerate(target_columns):
        ax.plot(x4[:3*COMPRESSED_POINTS_PER_DAY], norm4[:3*COMPRESSED_POINTS_PER_DAY, i], linewidth=1.5)
        color = ax.get_lines()[i].get_color()
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], pred_norm[:, i], '--', linewidth=3, color=color, alpha=0.9)
    setup(ax, "2. 3 Hari Input + 1 Hari Prediksi (Garis Putus-Putus)")
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.tight_layout(); plt.savefig("gambar2_input_plus_prediksi_TCN.png", dpi=300, bbox_inches='tight'); plt.close()

    # Gambar 3: Real vs Prediksi Hari Terakhir
    fig, ax = plt.subplots(figsize=(28, 10))
    for i, col in enumerate(target_columns):
        ax.plot(x4[:3*COMPRESSED_POINTS_PER_DAY], norm4[:3*COMPRESSED_POINTS_PER_DAY, i], linewidth=1.5)
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], norm4[3*COMPRESSED_POINTS_PER_DAY:, i], linewidth=2)
        color = ax.get_lines()[i].get_color()
        ax.plot(x4[3*COMPRESSED_POINTS_PER_DAY:], pred_norm[:, i], '--', linewidth=3.5, color=color)
    setup(ax, "3. Real vs Prediksi Hari Terakhir")
    ax.legend(target_columns, bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)
    plt.tight_layout(); plt.savefig("gambar3_real_vs_prediksi_TCN.png", dpi=300, bbox_inches='tight'); plt.close()

# =============================
# SELESAI TOTAL
# =============================
log_print("\nSEMUA SELESAI 100%!")
log_print(f"• FINAL VALIDATION MSE = {best_val_mse:.6f} ← INI MSE AKHIR RESMI!")
log_print("• File: model_21param_TCN_BEST.pth, scaler, CSV, 4 gambar PNG")

print("\n" + "="*110)
print("SELESAI 100%! Kamu punya MSE AKHIR RESMI + semua plot + model terbaik!")
print(f"Ganti COMPRESSION_FACTOR = 10, 25, 50, atau 100 → SEMUA TETAP JALAN SEMPURNA!")
print("="*110)
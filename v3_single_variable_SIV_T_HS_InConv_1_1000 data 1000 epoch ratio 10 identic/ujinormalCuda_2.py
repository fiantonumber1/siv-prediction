# =============================
# PYTORCH SINGLE VARIABLE + GPU + CHECKPOINTING (UJICOBA CEPAT)
# =============================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, time, timedelta

# ==================================================================
# KONTROL UTAMA (ubah di sini saja)
# ==================================================================
N_DUPLICATES       = 1000      # berapa hari identik
N_EPOCHS           = 1000
CHECKPOINT_INTERVAL= 50
CHECKPOINT_DIR     = "checkpoints_single"
VARIABLE           = 'SIV_T_HS_InConv_1'   # <--- SATU VARIABEL SAJA
# ==================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

folder_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "."

# =============================
# 1. BACA & PREPROCESS (single variable)
# =============================
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
csv_files = [f for f in csv_files if "hasil" not in os.path.basename(f).lower()]

if not csv_files:
    raise FileNotFoundError("Tidak ada file CSV!")

template_file = csv_files[0]
print(f"Template: {os.path.basename(template_file)} → duplikasi {N_DUPLICATES}x")

START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600
N_TAKE = 150_000
COMPRESSION_FACTOR = 10
POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR          # 15_000

def read_crop_compress(filepath):
    df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig', on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'), 
                                   format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['ts_date'])
    df[VARIABLE] = pd.to_numeric(df[VARIABLE].astype(str).str.replace(',', '.'), errors='coerce')
    df[VARIABLE] = df[VARIABLE].ffill().bfill()

    day = df['ts_date'].dt.date.iloc[0]
    start_dt = datetime.combine(day, START_TIME)
    end_dt   = datetime.combine(day, END_TIME)
    df = df[(df['ts_date'] >= start_dt) & (df['ts_date'] <= end_dt)]
    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)

    # Kompresi rata-rata tiap 10 data
    compressed = []
    ts_mid = []
    for i in range(POINTS_PER_DAY):
        s, e = i*10, (i+1)*10
        compressed.append(df[VARIABLE].iloc[s:e].mean())
        ts_mid.append(df['ts_date'].iloc[s + 5])
    return pd.DataFrame({VARIABLE: compressed, 'ts_date': ts_mid})

template = read_crop_compress(template_file)

# Duplikasi identik
dfs = []
for day_idx in range(N_DUPLICATES):
    df_day = template.copy()
    offset = timedelta(days=day_idx)
    df_day['ts_date'] = df_day['ts_date'].dt.normalize() + offset + \
                        (df_day['ts_date'] - df_day['ts_date'].dt.normalize())
    dfs.append(df_day)

print(f"Data siap: {N_DUPLICATES} hari identik ({POINTS_PER_DAY} titik/hari)")

# =============================
# 2. LABEL HEALTH (dummy tetap semua sehat karena data identik)
# =============================
health_labels = [0] * N_DUPLICATES                               # 0 = Sehat
status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}

# =============================
# 3. BUAT SEQUENCE
# =============================
WINDOW = 3 * POINTS_PER_DAY     # 3 hari input
FUTURE = POINTS_PER_DAY         # prediksi 1 hari

X, y_sig, y_stat = [], [], []
for i in range(len(dfs) - 3):
    seq = np.concatenate([dfs[i][VARIABLE].values,
                          dfs[i+1][VARIABLE].values,
                          dfs[i+2][VARIABLE].values])
    X.append(seq)
    y_sig.append(dfs[i+3][VARIABLE].values)
    y_stat.append(health_labels[i+3])

X = np.array(X, dtype=np.float32)[:, :, np.newaxis]      # (samples, WINDOW, 1)
y_sig = np.array(y_sig, dtype=np.float32)[:, :, np.newaxis]
y_stat = np.array(y_stat, dtype=np.int64)

# Scaling
scaler = MinMaxScaler(feature_range=(-0.2, 1.2))
X_flat = X.reshape(-1, 1)
y_flat = y_sig.reshape(-1, 1)
scaler.fit(np.concatenate([X_flat, y_flat]))

X_scaled = scaler.transform(X_flat).reshape(X.shape)
y_scaled = scaler.transform(y_flat).reshape(y_sig.shape)

X_tensor = torch.from_numpy(X_scaled).to(device)
y_sig_tensor = torch.from_numpy(y_scaled).to(device)
y_stat_tensor = torch.from_numpy(y_stat).to(device)

# Dataset
class SeqDataset(Dataset):
    def __init__(self, X, y_sig, y_stat):
        self.X, self.y_sig, self.y_stat = X, y_sig, y_stat
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_sig[idx], self.y_stat[idx]

dataset = SeqDataset(X_tensor, y_sig_tensor, y_stat_tensor)
loader = DataLoader(dataset, batch_size=4, shuffle=True)   # batch kecil biar cepat

# =============================
# 4. MODEL (Seq2Seq + Classification)
# =============================
class SingleVarSeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(1, 128, batch_first=True)
        self.decoder = nn.LSTM(128, 64, batch_first=True)
        self.out_sig = nn.Linear(64, 1)
        self.cls_fc1 = nn.Linear(128, 32)
        self.cls_out = nn.Linear(32, 3)          # 3 kelas

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.squeeze(0)

        # Decoder: repeat hidden state sebanyak FUTURE
        dec_in = h.unsqueeze(1).repeat(1, FUTURE, 1)
        dec_out, _ = self.decoder(dec_in)
        sig_pred = self.out_sig(dec_out)

        # Classification
        cls_h = torch.relu(self.cls_fc1(h))
        cls_pred = self.cls_out(cls_h)
        return sig_pred, cls_pred

model = SingleVarSeq2Seq().to(device)
opt = optim.Adam(model.parameters(), lr=0.001)
loss_mse = nn.MSELoss()
loss_ce  = nn.CrossEntropyLoss()

# =============================
# 5. CHECKPOINT (lanjut kalau putus)
# =============================
start_epoch = 1
checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "cp_*.pth"))
if checkpoint_files:
    epochs = [int(f.split('_')[-1].replace('.pth','')) for f in checkpoint_files]
    latest = max(epochs)
    cp_path = os.path.join(CHECKPOINT_DIR, f"cp_{latest}.pth")
    cp = torch.load(cp_path)
    model.load_state_dict(cp['model'])
    opt.load_state_dict(cp['opt'])
    start_epoch = latest + 1
    print(f"Lanjut dari epoch {start_epoch}")

# =============================
# 6. TRAINING
# =============================
if start_epoch <= N_EPOCHS:
    model.train()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        epoch_loss = 0.0
        for x_b, ysig_b, ystat_b in loader:
            opt.zero_grad()
            pred_sig, pred_cls = model(x_b)
            loss1 = loss_mse(pred_sig, ysig_b)
            loss2 = loss_ce(pred_cls, ystat_b)
            loss = loss1 + 3.0 * loss2
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if epoch % 20 == 0 or epoch == N_EPOCHS:
            print(f"Epoch {epoch:4d} | Loss {epoch_loss:.6f}")

        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            path = os.path.join(CHECKPOINT_DIR, f"cp_{epoch}.pth")
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, path)
            print(f"   → Checkpoint: {path}")

# =============================
# 7. PREDIKSI AKHIR
# =============================
model.eval()
with torch.no_grad():
    last_x = X_tensor[-1:].to(device)
    pred_scaled, cls_prob = model(last_x)
    pred_scaled = pred_scaled.cpu().numpy()[0]
    cls_prob = cls_prob.cpu().numpy()[0]

pred_real = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
pred_status = np.argmax(cls_prob)
confidence = np.max(cls_prob) * 100

print(f"\nPREDIKSI HARI TERAKHIR → {status_map[pred_status]} ({confidence:.1f}% confidence)")

# =============================
# 8. PLOT SEMUA (sama persis seperti sebelumnya)
# =============================
full_data = pd.concat(dfs, ignore_index=True)[VARIABLE].values
norm_all = scaler.transform(full_data.reshape(-1, 1)).flatten()

plt.figure(figsize=(20, 8))
plt.plot(norm_all, linewidth=0.9, label=VARIABLE)
day_lines = np.arange(0, len(full_data)+1, POINTS_PER_DAY)
for xl in day_lines[1:-1]: plt.axvline(xl, color='red', ls='--', alpha=0.8)
plt.title(f"{N_DUPLICATES} Hari Identik - {VARIABLE}")
plt.xlabel("Titik waktu")
plt.ylabel("Normalisasi")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plot_all_single_var.png", dpi=300)
plt.close()

# Simpan model + scaler
torch.save(model.state_dict(), "model_single_var.pth")
joblib.dump(scaler, "scaler_single_var.pkl")

print("\nSELESAI! File yang dihasilkan:")
print("   plot_all_single_var.png")
print("   model_single_var.pth")
print("   scaler_single_var.pkl")
print("   checkpoints_single/ → semua checkpoint")
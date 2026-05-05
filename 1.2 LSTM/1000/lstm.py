# =============================
# SEQ2SEQ + CLASSIFICATION 21 PARAMETER - FULL LSTM VERSION
# VERSI FINAL KHUSUS DATA REAL 100% (TIDAK ADA DUPLIKASI)
# + MSE FORECAST & ACCURACY KLASIFIKASI PER EPOCH
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
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
CHECKPOINT_DIR = "checkpoints_21param_LSTM"
LOG_FILE = "training_log_21param_LSTM.txt"

COMPRESSION_FACTOR = 1             # 100 = cepat | 50 = sedang | 25 = detail | 10 = super detail
# ==================================================================

# ================== AUTO-CALCULATED ==================
N_TAKE = 150_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR

WINDOW = 3 * COMPRESSED_POINTS_PER_DAY      # 3 hari input
FUTURE = COMPRESSED_POINTS_PER_DAY          # 1 hari prediksi

START_TIME = time(6, 0, 0)
END_TIME   = time(18, 16, 35)
N_DROP_FIRST = 3600
# =====================================================

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

# Kompresi data
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
    fault_cols = ['SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInvFltPres']
    for col in fault_cols:
        if col in df_day.columns and (df_day[col] == 1).any():
            return 1, "Pre-Anomali"
    return 0, "Sehat"

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
# LSTM MODEL (MULTI-TASK)
# =============================
class LSTM_MultiTask(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=False)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Decoder untuk forecasting (seq2seq)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_features * FUTURE)
        )
        
        # Classifier untuk health status
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)   # 3 kelas: Sehat, Pre-Anomali, Near-Fail
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Gunakan last hidden state atau global average pooling
        feature = lstm_out.mean(dim=1)          # shape: (batch, hidden_size)
        # Alternatif: feature = hn[-1]           # last layer hidden state
        
        sig_pred = self.decoder(feature).view(-1, FUTURE, n_features)
        stat_pred = self.classifier(feature)
        
        return sig_pred, stat_pred

model = LSTM_MultiTask(n_features=n_features, hidden_size=128, num_layers=3, dropout=0.3).to(device)

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
        print(f"Training SUDAH SELESAI di epoch {latest_epoch}.")

# =============================
# LOG FUNCTION
# =============================
def log_print(text):
    print(text)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

log_print(f"\n{'='*80}")
log_print(f"LSTM TRAINING DATA REAL 100% - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Total sample: {len(X_tensor)} | Epoch: {N_EPOCHS} | Batch: {BATCH_SIZE}")
log_print(f"COMPRESSION_FACTOR: {COMPRESSION_FACTOR}× → {COMPRESSED_POINTS_PER_DAY} pts/hari")
log_print(f"{'='*80}")

# =============================
# TRAINING LOOP
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
            torch.save({
                'epoch': epoch, 
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss
            }, cp_path)
            log_print(f"   → Checkpoint: {cp_path}")

    log_print("=== TRAINING SELESAI ===\n")
else:
    model.load_state_dict(torch.load(latest_cp, map_location=device)['model'])

# =============================
# PREDIKSI & SIMPAN HASIL
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

torch.save(model.state_dict(), "model_21param_LSTM_final.pth")
joblib.dump(scaler, "scaler_21param_LSTM.pkl")
log_print("Model & scaler disimpan")

# =============================
# PLOT (sama seperti sebelumnya)
# =============================
# ... (Bagian plot tetap sama persis seperti kode TCN kamu)
# Kamu bisa copy-paste bagian plot dari kode asli kamu mulai dari:
# df_all = pd.concat(compressed_dfs, ignore_index=True)  sampai akhir

# (Untuk menghemat ruang, saya tidak tulis ulang seluruh plot di sini.
#  Silakan copy bagian plot dari kode TCN asli kamu, hanya ganti nama file menjadi _LSTM)

# Contoh perubahan nama file plot:
# plt.savefig("plot_all_parameters_LSTM.png", dpi=300, bbox_inches='tight')
# plt.savefig("gambar1_4hari_real_LSTM.png", ...)
# dst.

# =============================
# SIMPAN HASIL PREDIKSI CSV
# =============================
result_df = pd.DataFrame(pred_signal, columns=target_columns)
result_df.insert(0, 'ts_date', compressed_dfs[-1]['ts_date'].values[-FUTURE:])
result_df['health_status_pred'] = status_map[pred_status]
result_df['confidence_percent'] = pred_confidence
result_df.to_csv("prediksi_hari_depan_21param_LSTM.csv", index=False)
log_print("prediksi_hari_depan_21param_LSTM.csv disimpan")

log_print("\nSEMUA SELESAI 100% - VERSI LSTM DATA REAL TANPA DUPLIKASI!")
print("\n" + "="*90)
print("SELESAI! Kode sudah diubah full ke LSTM")
print("Ganti COMPRESSION_FACTOR sesuai kebutuhan")
print("Log training tersimpan di training_log_21param_LSTM.txt")
print("="*90)
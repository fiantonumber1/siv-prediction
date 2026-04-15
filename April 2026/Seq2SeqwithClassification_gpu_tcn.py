# =============================
# SEQ2SEQ + CLASSIFICATION 21 PARAMETER - FULL TCN VERSION
# VERSI FINAL KHUSUS DATA REAL 100% (TIDAK ADA DUPLIKASI SAMA SEKALI)
# + COMPREHENSIVE METRICS (MSE, MAE, RMSE, MAPE, R², Precision, Recall, F1, ROC-AUC)
# + EARLY STOPPING (patience=100, monitor val loss, save best model)
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_auc_score
)
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

EARLY_STOPPING_PATIENCE = 100  # Stop jika val loss tidak membaik dalam 100 epoch

COMPRESSION_FACTOR = 1             # SATU-SATUNYA ANGKA YANG PERNAH KAMU UBAH
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
    # Membaca file CSV dengan pemisah titik koma (;) dan encoding utf-8-sig.
    df = pd.read_csv(filepath, encoding='utf-8-sig', sep=';', low_memory=False, on_bad_lines='skip')
    # Membersihkan nama kolom dan mengonversi tipe data.
    df.columns = [col.strip() for col in df.columns]
    # Memperbaiki format timestamp (ts_date) dan mengubahnya menjadi tipe datetime.
    df['ts_date'] = pd.to_datetime(df['ts_date'].astype(str).str.replace(',', '.'), 
                                   format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['ts_date'])
    # Mengonversi 21 parameter target menjadi numerik (mengganti koma dengan titik sebagai desimal).
    for col in target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = np.nan
    # Mengisi nilai hilang (missing values) pada 21 parameter menggunakan metode forward-fill diikuti backward-fill.
    df[target_columns] = df[target_columns].ffill().bfill()

    file_date = df['ts_date'].dt.date.iloc[0]
    start_dt = datetime.combine(file_date, START_TIME)
    end_dt = datetime.combine(file_date, END_TIME)
    # Pemotongan waktu operasional: hanya data dari pukul 06:00:00 hingga 18:16:35 yang dipertahankan.
    df = df[(df['ts_date'] >= start_dt) & (df['ts_date'] <= end_dt)]
    
    if len(df) < N_DROP_FIRST + N_TAKE * 0.8:
        return pd.DataFrame()
    # Membuang 3.600 baris awal (fase transien) dan menggunakan 150.000 baris selanjutnya sebagai data harian.
    df = df.iloc[N_DROP_FIRST:N_DROP_FIRST + N_TAKE].reset_index(drop=True)
    return df[['ts_date'] + target_columns]

# Proses kompresi data harian
compressed_dfs = []
# Ambil data harian yang sudah dipotong
for f in csv_files_sorted:
    df_raw = read_and_crop(f)
    if df_raw.empty:
        print(f"Skip {os.path.basename(f)} → data kurang")
        continue
    chunks, ts_mid = [], []
    for i in range(COMPRESSED_POINTS_PER_DAY):
        s, e = i * COMPRESSION_FACTOR, (i + 1) * COMPRESSION_FACTOR
        # Metode kompresi: rata-rata (mean) pada setiap segmen (chunk) sepanjang COMPRESSION_FACTOR titik.
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
    # Identifikasi kolom fault utama
    fault_cols = [
        'SIV_MajorBCFltPres',
        'SIV_MajorInputConvFltPres',
        'SIV_MajorInvFltPres',
    ]

    for col in fault_cols:
        # Deteksi keberadaan fault
        if col in df_day.columns and (df_day[col] == 1).any():
            #Pemberian label status
            return 1, "Pre-Anomali"
    # Pemberian label status
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
    # Gabungkan 3 hari berturut-turut sebagai input sequence
    seq = np.concatenate([df[target_columns].values for df in compressed_dfs[i:i+3]], axis=0)
    X_seq.append(seq)
    # Hari ke-4 sebagai target prediksi
    y_signal.append(compressed_dfs[i+3][target_columns].values)
    # Label status hari ke-4
    y_status.append(health_status[i+3])


# Konversi tipe data
# Konversi list ke array sekaligus ke float32 untuk efisiensi memori 
X_seq = np.array(X_seq, dtype=np.float32)
y_signal = np.array(y_signal, dtype=np.float32)
# Konversi list ke array pada label status
y_status = np.array(y_status, dtype=np.int64)

# Normalisasi
# Skala fitur ke rentang -0.1 hingga 1.1
scaler = MinMaxScaler(feature_range=(-0.1, 1.1))
# Fit dan transformasi data input dan target
X_scaled = scaler.fit_transform(X_seq.reshape(-1, n_features)).reshape(X_seq.shape)
y_scaled = scaler.transform(y_signal.reshape(-1, n_features)).reshape(y_signal.shape)

# Tensor
# Konversi data ke tensor PyTorch dan pindahkan ke device (CPU/GPU)
X_tensor = torch.FloatTensor(X_scaled).to(device)
y_sig_tensor = torch.FloatTensor(y_scaled).to(device)
y_stat_tensor = torch.LongTensor(y_status).to(device)

# DataLoader untuk batching dan shuffling data
class SeqDataset(Dataset):
    def __init__(self, X, y_sig, y_stat):
        self.X, self.y_sig, self.y_stat = X, y_sig, y_stat
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_sig[idx], self.y_stat[idx]

# Split data: 80% train, 20% validation
n_samples = len(X_tensor)
n_train = int(0.8 * n_samples)
indices = torch.randperm(n_samples)
train_idx, val_idx = indices[:n_train], indices[n_train:]

train_loader = DataLoader(SeqDataset(X_tensor[train_idx], y_sig_tensor[train_idx], y_stat_tensor[train_idx]),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(SeqDataset(X_tensor[val_idx], y_sig_tensor[val_idx], y_stat_tensor[val_idx]),
                        batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"\n[DATA SPLIT] Train: {n_train} samples | Validation: {n_samples - n_train} samples\n")

dataloader = train_loader  # For backward compatibility with existing code

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

# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0
best_epoch = 1
best_model_path = os.path.join(CHECKPOINT_DIR, "best_model_TCN.pth")

# =============================
# METRIC CALCULATION FUNCTIONS
# =============================
def calculate_forecasting_metrics(y_true, y_pred):
    """Calculate comprehensive forecasting metrics"""
    # Flatten for per-timestep metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Basic metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    # MAPE (handle division by zero)
    mask = y_true_flat != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = np.nan
    
    # Per-parameter metrics (averaged across timesteps)
    # Reshape to (samples, timesteps, features)
    y_true_3d = y_true.reshape(-1, FUTURE, n_features)
    y_pred_3d = y_pred.reshape(-1, FUTURE, n_features)
    
    # Per-feature MSE (averaged)
    mse_per_feature = []
    for i in range(n_features):
        mse_f = mean_squared_error(y_true_3d[:, :, i].flatten(), y_pred_3d[:, :, i].flatten())
        mse_per_feature.append(mse_f)
    avg_mse_features = np.mean(mse_per_feature)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Avg_MSE_Per_Feature': avg_mse_features
    }

def calculate_classification_metrics(y_true, y_pred, y_prob=None, n_classes=3):
    """Calculate comprehensive classification metrics"""
    # Basic metrics
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    # Precision, Recall, F1 (per class and macro)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    # ROC-AUC (one-vs-rest) - only if probabilities are provided
    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            pass
    
    return {
        'Accuracy': accuracy,
        'Precision_Macro': precision_macro,
        'Recall_Macro': recall_macro,
        'F1_Macro': f1_macro,
        'Precision_Weighted': precision_weighted,
        'Recall_Weighted': recall_weighted,
        'F1_Weighted': f1_weighted,
        'Precision_Per_Class': precision_per_class,
        'Recall_Per_Class': recall_per_class,
        'F1_Per_Class': f1_per_class,
        'Confusion_Matrix': cm,
        'ROC_AUC': roc_auc
    }

def format_metrics_report(metrics_dict, prefix="Val"):
    """Format metrics for logging"""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"COMPREHENSIVE METRICS REPORT - {prefix}")
    lines.append(f"{'='*80}")
    
    # Forecasting metrics
    if 'MSE' in metrics_dict:
        lines.append(f"\n📊 FORECASTING METRICS:")
        lines.append(f"  MSE:  {metrics_dict['MSE']:.6f}")
        lines.append(f"  RMSE: {metrics_dict['RMSE']:.6f}")
        lines.append(f"  MAE:  {metrics_dict['MAE']:.6f}")
        lines.append(f"  MAPE: {metrics_dict['MAPE']:.2f}%")
        lines.append(f"  R²:   {metrics_dict['R2']:.6f}")
        lines.append(f"  Avg MSE/Feature: {metrics_dict['Avg_MSE_Per_Feature']:.6f}")
    
    # Classification metrics
    if 'Accuracy' in metrics_dict:
        lines.append(f"\n🎯 CLASSIFICATION METRICS:")
        lines.append(f"  Accuracy: {metrics_dict['Accuracy']*100:.2f}%")
        lines.append(f"  Precision (Macro): {metrics_dict['Precision_Macro']:.4f}")
        lines.append(f"  Recall (Macro):    {metrics_dict['Recall_Macro']:.4f}")
        lines.append(f"  F1-Score (Macro):  {metrics_dict['F1_Macro']:.4f}")
        lines.append(f"  Precision (Weighted): {metrics_dict['Precision_Weighted']:.4f}")
        lines.append(f"  Recall (Weighted):    {metrics_dict['Recall_Weighted']:.4f}")
        lines.append(f"  F1-Score (Weighted):  {metrics_dict['F1_Weighted']:.4f}")
        
        if metrics_dict['ROC_AUC'] is not None:
            lines.append(f"  ROC-AUC (OvR): {metrics_dict['ROC_AUC']:.4f}")
        
        # Per-class metrics
        lines.append(f"\n📈 PER-CLASS METRICS:")
        status_map = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
        for i in range(len(metrics_dict['Precision_Per_Class'])):
            lines.append(f"  Class {i} ({status_map.get(i, 'Unknown')}):")
            lines.append(f"    Precision: {metrics_dict['Precision_Per_Class'][i]:.4f}")
            lines.append(f"    Recall:    {metrics_dict['Recall_Per_Class'][i]:.4f}")
            lines.append(f"    F1-Score:  {metrics_dict['F1_Per_Class'][i]:.4f}")
        
        # Confusion Matrix
        lines.append(f"\n🔲 CONFUSION MATRIX:")
        lines.append(f"  Predicted →")
        for i, row in enumerate(metrics_dict['Confusion_Matrix']):
            true_label = status_map.get(i, f"Class {i}")
            lines.append(f"  Actual: {true_label:12s} | {row}")
    
    lines.append(f"{'='*80}\n")
    return "\n".join(lines)

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
# TRAINING LOOP + MSE & ACCURACY PER EPOCH + EARLY STOPPING
# =============================
if start_epoch <= N_EPOCHS:
    model.train()
    for epoch in range(start_epoch, N_EPOCHS + 1):
        # --- TRAINING PHASE ---
        model.train()
        total_loss = total_mse = 0.0
        total_correct = total_samples = 0
        train_y_true = []
        train_y_pred = []
        train_sig_true = []
        train_sig_pred = []

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
            
            # Collect for metrics
            train_y_true.extend(y_stat.cpu().numpy())
            train_y_pred.extend(pred.cpu().numpy())
            train_sig_true.extend(y_sig.cpu().numpy())
            train_sig_pred.extend(sig_pred.detach().cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        acc = 100.0 * total_correct / total_samples

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = val_mse = 0.0
        val_correct = val_samples = 0
        val_y_true = []
        val_y_pred = []
        val_y_prob = []
        val_sig_true = []
        val_sig_pred = []

        with torch.no_grad():
            for x, y_sig, y_stat in val_loader:
                sig_pred, stat_pred = model(x)
                loss_sig = criterion_mse(sig_pred, y_sig)
                loss_cls = criterion_ce(stat_pred, y_stat)
                loss = loss_sig + 3.5 * loss_cls

                val_loss += loss.item()
                val_mse += loss_sig.item()
                _, pred = torch.max(stat_pred, 1)
                val_correct += (pred == y_stat).sum().item()
                val_samples += y_stat.size(0)
                
                # Collect for metrics
                val_y_true.extend(y_stat.cpu().numpy())
                val_y_prob.extend(torch.softmax(stat_pred, dim=1).detach().cpu().numpy())
                val_y_pred.extend(pred.cpu().numpy())
                val_sig_true.extend(y_sig.cpu().numpy())
                val_sig_pred.extend(sig_pred.detach().cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        val_acc = 100.0 * val_correct / val_samples

        # Learning rate scheduler
        scheduler.step(avg_val_loss)

        # Log training and validation metrics
        log_line = f"Epoch {epoch:4d}/{N_EPOCHS} | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | MSE(Forecast): {avg_mse:.7f} | Val MSE: {avg_val_mse:.7f} | Acc(Class): {acc:6.2f}% | Val Acc: {val_acc:6.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}"
        print(log_line)
        log_print(log_line)
        
        # Calculate and log comprehensive metrics every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            # Training metrics
            train_arr = np.array(train_y_true)
            train_pred_arr = np.array(train_y_pred)
            train_cls_metrics = calculate_classification_metrics(
                train_arr, train_pred_arr, n_classes=3
            )
            
            train_sig_arr = np.array(train_sig_true)
            train_sig_pred_arr = np.array(train_sig_pred)
            train_forecast_metrics = calculate_forecasting_metrics(
                train_sig_arr, train_sig_pred_arr
            )
            
            log_print(format_metrics_report(train_forecast_metrics, "Train - Forecast"))
            log_print(format_metrics_report(train_cls_metrics, "Train - Classification"))
            
            # Validation metrics
            val_arr = np.array(val_y_true)
            val_pred_arr = np.array(val_y_pred)
            val_prob_arr = np.array(val_y_prob)
            val_cls_metrics = calculate_classification_metrics(
                val_arr, val_pred_arr, val_prob_arr, n_classes=3
            )
            
            val_sig_arr = np.array(val_sig_true)
            val_sig_pred_arr = np.array(val_sig_pred)
            val_forecast_metrics = calculate_forecasting_metrics(
                val_sig_arr, val_sig_pred_arr
            )
            
            log_print(format_metrics_report(val_forecast_metrics, "Val - Forecast"))
            log_print(format_metrics_report(val_cls_metrics, "Val - Classification"))

        # --- EARLY STOPPING CHECK ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }, best_model_path)
            if epoch % 10 == 0:
                log_print(f"   ✓ New best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1

        # Regular checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0 or epoch == N_EPOCHS:
            cp_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'loss': avg_loss}, cp_path)
            log_print(f"   → Checkpoint: {cp_path}")

        # Early stopping trigger
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            log_print(f"\n{'='*80}")
            log_print(f"EARLY STOPPING TRIGGERED at epoch {epoch}!")
            log_print(f"Best epoch: {best_epoch} | Best Val Loss: {best_val_loss:.6f}")
            log_print(f"Training stopped after {epoch - start_epoch + 1} epochs (patience={EARLY_STOPPING_PATIENCE})")
            log_print(f"{'='*80}\n")
            break

        # Switch back to train mode
        model.train()

    log_print("=== TRAINING SELESAI ===\n")
    
    # Load best model for prediction
    if os.path.exists(best_model_path):
        log_print(f"Loading best model from epoch {best_epoch} (Val Loss: {best_val_loss:.6f})")
        best_cp = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_cp['model'])
    else:
        log_print("Warning: No best model found, using last checkpoint")
    
    # Final comprehensive evaluation on full dataset
    log_print("\n" + "="*80)
    log_print("FINAL COMPREHENSIVE EVALUATION ON FULL DATASET")
    log_print("="*80)
    
    model.eval()
    with torch.no_grad():
        full_sig_pred, full_stat_pred = model(X_tensor)
        full_stat_prob = torch.softmax(full_stat_pred, dim=1)
        
        # Convert to numpy
        full_y_true_stat = y_stat_tensor.cpu().numpy()
        full_y_pred_stat = full_stat_pred.argmax(dim=1).cpu().numpy()
        full_y_prob_stat = full_stat_prob.cpu().numpy()
        
        full_y_true_sig = y_sig_tensor.cpu().numpy()
        full_y_pred_sig = full_sig_pred.cpu().numpy()
        
        # Classification metrics
        final_cls_metrics = calculate_classification_metrics(
            full_y_true_stat, full_y_pred_stat, full_y_prob_stat, n_classes=3
        )
        
        # Forecasting metrics
        final_forecast_metrics = calculate_forecasting_metrics(
            full_y_true_sig, full_y_pred_sig
        )
        
        log_print(format_metrics_report(final_forecast_metrics, "FULL DATASET - Forecast"))
        log_print(format_metrics_report(final_cls_metrics, "FULL DATASET - Classification"))
        
        # Save classification report to file
        cls_report = classification_report(full_y_true_stat, full_y_pred_stat, 
                                          target_names=['Sehat', 'Pre-Anomali', 'Near-Fail'],
                                          zero_division=0)
        report_file = os.path.join(folder_path, "classification_report_final.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FINAL CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Model: TCN Multi-Task (Best from epoch {best_epoch})\n")
            f.write(f"Validation Loss: {best_val_loss:.6f}\n")
            f.write("="*80 + "\n\n")
            f.write(cls_report)
            f.write("\n" + "="*80 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("="*80 + "\n")
            cm = final_cls_metrics['Confusion_Matrix']
            status_map_text = {0: "Sehat", 1: "Pre-Anomali", 2: "Near-Fail"}
            f.write(f"{'Actual/Predicted':<20}")
            for j in range(3):
                f.write(f"{status_map_text[j]:<15}")
            f.write("\n")
            for i, row in enumerate(cm):
                f.write(f"{status_map_text[i]:<20}")
                for val in row:
                    f.write(f"{val:<15}")
                f.write("\n")
        log_print(f"Classification report saved to: {report_file}")
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

# Save model and scaler
torch.save(model.state_dict(), "model_21param_TCN_final.pth")
joblib.dump(scaler, "scaler_21param_TCN.pkl")
log_print("Model & scaler disimpan")

# Also save the best model separately
if os.path.exists(best_model_path):
    log_print(f"Best model (epoch {best_epoch}, val loss {best_val_loss:.6f}) saved as: {best_model_path}")

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
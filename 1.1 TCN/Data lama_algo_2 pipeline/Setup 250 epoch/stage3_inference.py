# =============================
# STAGE 3 — INFERENCE PIPELINE
# CSV VERSION
# =============================

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, time
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# =========================================================
# AUTO AMBIL 3 CSV BERDASARKAN ALFABET
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_files = sorted([
    f for f in glob.glob(os.path.join(BASE_DIR, "*.csv"))
    if "inference" not in os.path.basename(f).lower()
])

print("\n[DEBUG] CSV ditemukan:")
for f in csv_files:
    print("  ", os.path.basename(f))

if len(csv_files) < 3:
    raise ValueError(
        f"Minimal harus ada 3 file CSV!\n"
        f"Ditemukan: {len(csv_files)}"
    )

CSV_DAY_A = csv_files[0]
CSV_DAY_B = csv_files[1]
CSV_DAY_C = csv_files[2]

print("\n[Inference] Menggunakan file:")
print(f"  Day A : {os.path.basename(CSV_DAY_A)}")
print(f"  Day B : {os.path.basename(CSV_DAY_B)}")
print(f"  Day C : {os.path.basename(CSV_DAY_C)}")

# =========================================================
# CONFIG
# =========================================================
COMPRESSION_FACTOR         = 1
N_TAKE                     = 150_000
COMPRESSED_POINTS_PER_DAY = N_TAKE // COMPRESSION_FACTOR
FUTURE                     = COMPRESSED_POINTS_PER_DAY

START_TIME   = time(6, 0, 0)
END_TIME     = time(18, 16, 35)
N_DROP_FIRST = 3600

N_FEATURES  = 21
N_CH        = 96
N_STATS     = 4
INPUT_DIM_2 = N_FEATURES * N_STATS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n[Inference] Device: {device}")

# =========================================================
# TARGET COLUMNS
# =========================================================
target_columns = [
    'SIV_T_HS_InConv_1',
    'SIV_T_HS_InConv_2',
    'SIV_T_HS_Inv_1',
    'SIV_T_HS_Inv_2',
    'SIV_T_Container',
    'SIV_I_L1',
    'SIV_I_L2',
    'SIV_I_L3',
    'SIV_I_Battery',
    'SIV_I_DC_In',
    'SIV_U_Battery',
    'SIV_U_DC_In',
    'SIV_U_DC_Out',
    'SIV_U_L1',
    'SIV_U_L2',
    'SIV_U_L3',
    'SIV_InConv_InEnergy',
    'SIV_Output_Energy',
    'PLC_OpenACOutputCont',
    'PLC_OpenInputCont',
    'SIV_DevIsAlive',
]

# =========================================================
# CEK FILE MODEL
# =========================================================
required = [
    "model_stage1_forecaster.pth",
    "scaler_stage1.pkl",
    "model_stage2_classifier.pth",
    "scaler_stage2.pkl"
]

for f in required:

    if not os.path.exists(f):

        raise FileNotFoundError(
            f"\nFile '{f}' tidak ditemukan!\n"
            f"Pastikan training stage1 dan stage2 selesai."
        )

# =========================================================
# LOAD SCALER
# =========================================================
scaler1 = joblib.load("scaler_stage1.pkl")
scaler2 = joblib.load("scaler_stage2.pkl")

print("[Inference] Scaler loaded")

# =========================================================
# MODEL
# =========================================================
class CausalConv1d(nn.Module):

    def __init__(self, in_ch, out_ch, ks, dilation=1):

        super().__init__()

        self.pad = (ks - 1) * dilation

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            ks,
            padding=self.pad,
            dilation=dilation
        )

    def forward(self, x):

        o = self.conv(x)

        return o[:, :, :-self.pad] if self.pad > 0 else o


class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ks=3, dilation=1, dropout=0.3):

        super().__init__()

        self.c1 = CausalConv1d(in_ch, out_ch, ks, dilation)
        self.n1 = nn.BatchNorm1d(out_ch)
        self.r1 = nn.ReLU()
        self.d1 = nn.Dropout(dropout)

        self.c2 = CausalConv1d(out_ch, out_ch, ks, dilation)
        self.n2 = nn.BatchNorm1d(out_ch)
        self.r2 = nn.ReLU()
        self.d2 = nn.Dropout(dropout)

        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):

        r = self.skip(x)

        o = self.d1(self.r1(self.n1(self.c1(x))))
        o = self.d2(self.r2(self.n2(self.c2(o))))

        return o + r


class TCNForecaster(nn.Module):

    def __init__(
        self,
        n_features=21,
        n_ch=96,
        ks=3,
        n_blocks=7,
        dropout=0.3
    ):

        super().__init__()

        dilations = [1, 2, 4, 8, 16, 32, 64]

        layers = []
        in_ch = n_features

        for i in range(n_blocks):

            layers.append(
                ResidualBlock(
                    in_ch,
                    n_ch,
                    ks,
                    dilations[i],
                    dropout
                )
            )

            in_ch = n_ch

        self.tcn = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dec = nn.Sequential(
            nn.Linear(n_ch, n_ch),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(
                n_ch,
                n_features * FUTURE
            )
        )

    def forward(self, x):

        ctx = self.pool(
            self.tcn(x.transpose(1, 2))
        ).squeeze(-1)

        pred = self.dec(ctx).view(
            -1,
            FUTURE,
            N_FEATURES
        )

        return pred, ctx


class MLPClassifier(nn.Module):

    def __init__(self, input_dim=84, dropout=0.3):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 3)
        )

    def forward(self, x):

        return self.net(x)

# =========================================================
# LOAD MODEL
# =========================================================
tcn_model = TCNForecaster(N_FEATURES, N_CH).to(device)

tcn_model.load_state_dict(
    torch.load(
        "model_stage1_forecaster.pth",
        map_location=device
    )
)

tcn_model.eval()

mlp_model = MLPClassifier(INPUT_DIM_2).to(device)

mlp_model.load_state_dict(
    torch.load(
        "model_stage2_classifier.pth",
        map_location=device
    )
)

mlp_model.eval()

print("[Inference] Model TCN dan MLP loaded")

# =========================================================
# READ CSV
# =========================================================
def read_csv_day(filepath):

    print(f"\n  Membaca: {os.path.basename(filepath)}")

    # =====================================================
    # AUTO DETECT SEPARATOR
    # =====================================================
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:

        first_line = f.readline()

    comma_count    = first_line.count(',')
    semicolon_count = first_line.count(';')

    sep = ';' if semicolon_count > comma_count else ','

    print(f"    Separator: '{sep}'")

    # =====================================================
    # READ CSV
    # =====================================================
    try:

        df = pd.read_csv(
            filepath,
            sep=sep,
            engine='python',
            on_bad_lines='skip'
        )

    except Exception as e:

        raise ValueError(
            f"\nGagal membaca CSV:\n"
            f"{filepath}\n\n{str(e)}"
        )

    print(f"    Shape awal: {df.shape}")

    # =====================================================
    # NORMALISASI NAMA KOLOM
    # =====================================================
    df.columns = [str(c).strip() for c in df.columns]

    # =====================================================
    # CARI TIMESTAMP
    # =====================================================
    ts_col = None

    for c in df.columns:

        c_lower = c.lower()

        if (
            'date' in c_lower or
            'time' in c_lower or
            'timestamp' in c_lower or
            'ts' in c_lower
        ):

            ts_col = c
            break

    if ts_col is None:
        ts_col = df.columns[0]

    print(f"    Timestamp column: {ts_col}")

    # =====================================================
    # PARSE DATETIME
    # =====================================================
    df['ts_date'] = pd.to_datetime(
        df[ts_col].astype(str).str.replace(',', '.'),
        errors='coerce'
    )

    df = df.dropna(subset=['ts_date'])

    print(f"    Setelah parse datetime: {len(df)} rows")

    # =====================================================
    # AMBIL SENSOR
    # =====================================================
    for col in target_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(',', '.'),
                errors='coerce'
            )

        else:

            print(f"    WARNING: '{col}' tidak ditemukan")
            df[col] = np.nan

    df[target_columns] = df[target_columns].ffill().bfill()

    # =====================================================
    # CROP JAM OPERASI
    # =====================================================
    date0 = df['ts_date'].dt.date.iloc[0]

    df = df[
        (df['ts_date'] >= datetime.combine(date0, START_TIME)) &
        (df['ts_date'] <= datetime.combine(date0, END_TIME))
    ]

    print(f"    Setelah crop: {len(df)} rows")

    # =====================================================
    # VALIDASI
    # =====================================================
    min_required = int(N_DROP_FIRST + N_TAKE * 0.5)

    if len(df) < min_required:

        raise ValueError(
            f"\nData terlalu sedikit.\n"
            f"Rows: {len(df)}\n"
            f"Minimal: {min_required}"
        )

    # =====================================================
    # DROP AWAL
    # =====================================================
    df = df.iloc[
        N_DROP_FIRST:N_DROP_FIRST + N_TAKE
    ].reset_index(drop=True)

    print(f"    Setelah drop awal: {len(df)} rows")

    # =====================================================
    # KOMPRESI
    # =====================================================
    chunks = []
    ts_mid = []

    n_pts = min(
        COMPRESSED_POINTS_PER_DAY,
        len(df) // COMPRESSION_FACTOR
    )

    for i in range(n_pts):

        s = i * COMPRESSION_FACTOR
        e = (i + 1) * COMPRESSION_FACTOR

        chunk = df[target_columns].iloc[s:e]

        chunks.append(chunk.mean())

        mid_idx = min(
            s + COMPRESSION_FACTOR // 2,
            len(df) - 1
        )

        ts_mid.append(
            df['ts_date'].iloc[mid_idx]
        )

    df_c = pd.DataFrame(
        chunks,
        columns=target_columns
    )

    df_c.insert(0, 'ts_date', ts_mid)

    print(f"    → {len(df_c)} titik terkompresi")

    return df_c

# =========================================================
# LOAD 3 CSV
# =========================================================
print("\n[Inference] Membaca 3 file CSV...")

try:

    df_A = read_csv_day(CSV_DAY_A)
    df_B = read_csv_day(CSV_DAY_B)
    df_C = read_csv_day(CSV_DAY_C)

except Exception as e:

    print("\nERROR:")
    print(str(e))
    raise

# =========================================================
# STAGE 1
# =========================================================
print("\n[Stage 1] TCN Forecaster")

seq_raw = np.concatenate([
    df_A[target_columns].values,
    df_B[target_columns].values,
    df_C[target_columns].values,
], axis=0).astype(np.float32)

seq_scaled = scaler1.transform(
    seq_raw.reshape(-1, N_FEATURES)
).reshape(seq_raw.shape)

x_tensor = torch.FloatTensor(
    seq_scaled
).unsqueeze(0).to(device)

with torch.no_grad():

    pred_scaled, ctx = tcn_model(x_tensor)

    pred_scaled = pred_scaled.cpu().numpy()[0]

pred_sensor = scaler1.inverse_transform(
    pred_scaled.reshape(-1, N_FEATURES)
).reshape(FUTURE, N_FEATURES)

print(f"  → Shape prediksi: {pred_sensor.shape}")

# =========================================================
# STAGE 2
# =========================================================
print("\n[Stage 2] MLP Classifier")

feat = np.concatenate([

    pred_sensor.mean(axis=0),
    pred_sensor.std(axis=0),
    pred_sensor.max(axis=0),
    pred_sensor.min(axis=0),

]).astype(np.float32)

feat_scaled = scaler2.transform(
    feat.reshape(1, -1)
)

feat_tensor = torch.FloatTensor(
    feat_scaled
).to(device)

with torch.no_grad():

    logits = mlp_model(feat_tensor)

    prob = torch.softmax(
        logits,
        dim=1
    ).cpu().numpy()[0]

pred_status = int(np.argmax(prob))

pred_confidence = float(
    prob[pred_status] * 100
)

status_map = {
    0: "Sehat",
    1: "Pre-Anomali",
    2: "Near-Fail"
}

status_color = {
    0: "HIJAU",
    1: "KUNING",
    2: "MERAH"
}

# =========================================================
# OUTPUT
# =========================================================
print("\n" + "="*60)
print("HASIL PREDIKSI HARI D")
print("="*60)

print(
    f"Input : "
    f"{os.path.basename(CSV_DAY_A)} + "
    f"{os.path.basename(CSV_DAY_B)} + "
    f"{os.path.basename(CSV_DAY_C)}"
)

print(
    f"Status : "
    f"[{status_color[pred_status]}] "
    f"{status_map[pred_status]}"
)

print(f"Confidence : {pred_confidence:.2f}%")

print(f"Prob Sehat       : {prob[0]*100:.2f}%")
print(f"Prob Pre-Anomali : {prob[1]*100:.2f}%")
print(f"Prob Near-Fail   : {prob[2]*100:.2f}%")

print("="*60)

# =========================================================
# PREVIEW
# =========================================================
print("\nPreview prediksi sensor:")

preview = pd.DataFrame(
    pred_sensor[:5, :5],
    columns=target_columns[:5]
)

print(
    preview.round(3).to_string(index=False)
)

# =========================================================
# SAVE SENSOR CSV
# =========================================================
result_sensor = pd.DataFrame(
    pred_sensor,
    columns=target_columns
)

result_sensor.insert(
    0,
    'timestep',
    range(len(result_sensor))
)

result_sensor.to_csv(
    "inference_prediksi_sensor.csv",
    index=False
)

print("\ninference_prediksi_sensor.csv disimpan")

# =========================================================
# SAVE STATUS CSV
# =========================================================
result_status = pd.DataFrame([{

    'input_day_a': os.path.basename(CSV_DAY_A),
    'input_day_b': os.path.basename(CSV_DAY_B),
    'input_day_c': os.path.basename(CSV_DAY_C),

    'health_status': status_map[pred_status],
    'status_code': pred_status,

    'confidence_pct': round(pred_confidence, 2),

    'prob_sehat_pct': round(float(prob[0]*100), 2),
    'prob_pre_anomali_pct': round(float(prob[1]*100), 2),
    'prob_near_fail_pct': round(float(prob[2]*100), 2),

    'generated_at': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'
    )

}])

result_status.to_csv(
    "inference_health_status.csv",
    index=False
)

print("inference_health_status.csv disimpan")

print("\nSELESAI!")
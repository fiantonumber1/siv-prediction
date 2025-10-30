# klasifikasi_model.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import io

# Daftar kolom fitur dan label
TARGET_COLUMNS = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive'
]

LABEL_COLUMNS = [
    'SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres',
    'Ux_SIV_MajorBCFltPres', 'Ux_SIV_MajorInputConvFltPres', 'Ux_SIV_MajorInverterFltPres'
]

LABEL_DEFINITIONS = {
    'SIV_MajorBCFltPres': "Major failure pada battery charger (device stopped)",
    'SIV_MajorInputConvFltPres': "Major failure pada input converter (device stopped)",
    'SIV_MajorInverterFltPres': "Major failure pada inverter (device stopped)",
    'Ux_SIV_MajorBCFltPres': "UX Major failure pada battery charger",
    'Ux_SIV_MajorInputConvFltPres': "UX Major failure pada input converter",
    'Ux_SIV_MajorInverterFltPres': "UX Major failure pada inverter"
}

def preprocess_data(df_stream):
    content = df_stream.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(content), sep=';', dtype=str)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.replace(',', '.', regex=False))
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def train_and_save(df_stream, model_name, model_dir):
    df = preprocess_data(df_stream)
    
    missing = [col for col in TARGET_COLUMNS + LABEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {missing}")
    
    df = df[TARGET_COLUMNS + LABEL_COLUMNS].dropna()
    if len(df) == 0:
        raise ValueError("Tidak ada data setelah pembersihan!")
    
    X = df[TARGET_COLUMNS].values
    y = df[LABEL_COLUMNS].astype(int).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(len(LABEL_COLUMNS), activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_scaled, y, epochs=50, batch_size=64, validation_split=0.2,
              callbacks=[early_stop], verbose=0)

    # Simpan model dan scaler
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_clf.h5")
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Evaluasi
    loss, acc = model.evaluate(X_scaled, y, verbose=0)
    return {"accuracy": acc, "loss": loss, "data_count": len(df)}

def predict_with_model(df_stream, model_name, model_dir):
    df = preprocess_data(df_stream)
    
    missing = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Fitur tidak ditemukan: {missing}")
    
    X = df[TARGET_COLUMNS].values
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    model_path = os.path.join(model_dir, f"{model_name}_clf.h5")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model atau scaler tidak ditemukan!")

    scaler = joblib.load(scaler_path)
    from tensorflow.keras.models import load_model
    model = load_model(model_path)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled, verbose=0)
    y_pred_binary = (y_pred > 0.5).astype(int)

    results = []
    for i in range(min(10, len(df))):  # Tampilkan 10 baris pertama
        row = {label: int(y_pred_binary[i][j]) for j, label in enumerate(LABEL_COLUMNS)}
        row['definitions'] = {k: LABEL_DEFINITIONS[k] for k, v in row.items() if v == 1 and k in LABEL_DEFINITIONS}
        results.append(row)

    return results
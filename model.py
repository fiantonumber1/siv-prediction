# model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TARGET_COLUMNS = [
    'SIV_T_HS_InConv_1', 'SIV_T_HS_InConv_2', 'SIV_T_HS_Inv_1', 'SIV_T_HS_Inv_2', 'SIV_T_Container',
    'SIV_I_L1', 'SIV_I_L2', 'SIV_I_L3', 'SIV_I_Battery', 'SIV_I_DC_In',
    'SIV_U_Battery', 'SIV_U_DC_In', 'SIV_U_DC_Out', 'SIV_U_L1', 'SIV_U_L2', 'SIV_U_L3',
    'SIV_InConv_InEnergy', 'SIV_Output_Energy',
    'PLC_OpenACOutputCont', 'PLC_OpenInputCont', 'SIV_DevIsAlive'
]

WINDOW_SIZE = 20

def preprocess_data(file_content):
    df = pd.read_csv(file_content, sep=';', converters={i: lambda x: str(x).replace(',', '.') for i in range(100)})
    df.columns = df.columns.str.strip()
    df = df.replace(',', '.', regex=True)
    df['ts_date'] = pd.to_datetime(df['ts_date'], errors='coerce')

    missing = [col for col in TARGET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang: {missing}")

    df_sel = df[TARGET_COLUMNS].copy()
    df_sel = df_sel.apply(pd.to_numeric, errors='coerce').ffill().bfill().dropna()
    return df_sel

def create_sequences(data, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def train_and_save(df, model_name, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = create_sequences(scaled)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential([LSTM(64, input_shape=(WINDOW_SIZE, len(TARGET_COLUMNS))), Dense(len(TARGET_COLUMNS))])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=1)

    # Simpan model + scaler
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Evaluasi
    pred = model.predict(X_test)
    pred_inv = scaler.inverse_transform(pred)
    y_inv = scaler.inverse_transform(y_test)
    mae = mean_absolute_error(y_inv, pred_inv)
    mse = mean_squared_error(y_inv, pred_inv)
    rmse = np.sqrt(mse)

    return {"mae": round(mae, 4), "mse": round(mse, 4), "rmse": round(rmse, 4), "model_path": model_path}

def predict_with_model(df_new, model_name, model_dir="models"):
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.pkl")
    model_path = os.path.join(model_dir, f"{model_name}.h5")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model atau scaler tidak ditemukan!")

    from tensorflow.keras.models import load_model
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    df_clean = preprocess_data(df_new)
    scaled = scaler.transform(df_clean)

    # Prediksi satu langkah dari data terakhir
    last_seq = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, len(TARGET_COLUMNS))
    pred_scaled = model.predict(last_seq)
    pred = scaler.inverse_transform(pred_scaled)[0]
    pred_dict = {col: round(val, 2) for col, val in zip(TARGET_COLUMNS, pred)}

    # Plot tren: aktual vs prediksi (rolling)
    actual = df_clean.iloc[-100:][TARGET_COLUMNS[0]].values  # contoh: kolom pertama
    pred_trend = []
    current_seq = scaled[-WINDOW_SIZE:].copy()
    for _ in range(10):
        p = model.predict(current_seq.reshape(1, WINDOW_SIZE, len(TARGET_COLUMNS)))[0]
        pred_trend.append(scaler.inverse_transform(p.reshape(1, -1))[0][0])
        current_seq = np.vstack([current_seq[1:], p])

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode='lines', name='Aktual'))
    fig.add_trace(go.Scatter(y=[None]*90 + pred_trend, mode='lines', name='Prediksi', line=dict(dash='dot')))
    fig.update_layout(title=f"Tren Prediksi vs Aktual ({TARGET_COLUMNS[0]})", xaxis_title="Waktu", yaxis_title="Nilai")
    
    return pred_dict, fig.to_html(full_html=False)
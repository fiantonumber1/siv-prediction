# integrated_model.py
import os
import numpy as np
import joblib
import io
from tensorflow.keras.models import load_model
from forecast_model import predict_with_model
from klasifikasi_model import TARGET_COLUMNS, predict_manual

def predict_integrated(file_content, forecast_model_name, clf_model_name, model_dir="models", steps=1):
    """
    Forecast multi-step → Klasifikasi tiap langkah.
    """
    # --- 1. Forecast ---
    df_stream = io.StringIO(file_content.decode('utf-8'))
    pred_dict, plot_html = predict_with_model(df_stream, forecast_model_name, model_dir, steps=steps)

    # --- 2. Klasifikasi untuk SETIAP langkah ---
    classification_per_step = {}
    for step_key, pred_values_dict in pred_dict.items():
        pred_values = [pred_values_dict[col] for col in TARGET_COLUMNS]
        clf_result = predict_manual(pred_values, clf_model_name, model_dir)
        classification_per_step[step_key] = clf_result

    return {
        "forecast": {
            "steps": steps,
            "prediction_dict": pred_dict
        },
        "classification_per_step": classification_per_step,
        "plot_html": plot_html
    }
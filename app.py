# app.py
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
import io
from forecast_model import preprocess_data, train_and_save, predict_with_model
# app.py (tambahkan ini)
# Import klasifikasi + TARGET_COLUMNS
from klasifikasi_model import (
    train_and_save as clf_train,
    predict_with_model as clf_predict,
    predict_manual,
    TARGET_COLUMNS,    
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_DIR'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)


# === HALAMAN UTAMA (DASHBOARD) ===
@app.route('/')
def dashboard():
    return render_template('dashboard.html')  # Halaman awal dengan tombol Forecast/Klasifikasi


# === HALAMAN FORECAST ===
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    models = [f.split('_scaler')[0] for f in os.listdir(app.config['MODEL_DIR']) if f.endswith('_scaler.pkl')]
    results = None
    error = None
    plot_html = None
    prediction = None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'train':
            file = request.files['file']
            model_name = request.form['model_name'].strip()
            if not file or not model_name:
                error = "File dan nama model wajib diisi!"
            else:
                try:
                    df = preprocess_data(io.StringIO(file.stream.read().decode('utf-8')))
                    metrics = train_and_save(df, model_name, app.config['MODEL_DIR'])
                    results = {
                        "model_name": model_name,
                        "mae": metrics['mae'],
                        "rmse": metrics['rmse'],
                        "r2": metrics['r2'],
                        "mape": metrics['mape'],
                        "r2_per_col": metrics['r2_per_col'],
                        "mape_per_col": metrics['mape_per_col']
                    }
                except Exception as e:
                    error = str(e)

        elif action == 'predict':
            file = request.files['file']
            model_name = request.form['model_select']
            steps = int(request.form.get('steps', 1))
            if not file or not model_name:
                error = "File dan model wajib dipilih!"
            else:
                try:
                    df_stream = io.StringIO(file.stream.read().decode('utf-8'))
                    prediction, plot_html = predict_with_model(df_stream, model_name, app.config['MODEL_DIR'], steps=steps)
                except Exception as e:
                    error = str(e)

    return render_template('forecast.html', 
                           models=models, 
                           results=results, 
                           error=error, 
                           plot_html=plot_html, 
                           prediction=prediction)


# === HALAMAN KLASIFIKASI (kosong dulu, bisa dikembangkan nanti) ===
@app.route('/klasifikasi', methods=['GET', 'POST'])
def klasifikasi():
    # Daftar model klasifikasi
    clf_models = [f.split('_clf')[0] for f in os.listdir(app.config['MODEL_DIR']) 
                  if f.endswith('_clf.h5')]
    
    results = None
    error = None
    prediction = None
    default_values = [
        43.98, 43.98, 56.09, 54.08, 54.95,
        73.87, 71.09, 72.15, -0.82, 59.98,
        112.93, 816.39, 126.12, 220.27, 219.72, 219.81,
        938.66, 815.11, 0.00, 0.00, 35472.18
    ]

    if request.method == 'POST':
        action = request.form.get('action')

        # === TRAIN DARI CSV (tetap ada) ===
        if action == 'train_clf':
            file = request.files['file']
            model_name = request.form['model_name'].strip()
            if not file or not model_name:
                error = "File dan nama model wajib diisi!"
            else:
                try:
                    df_stream = io.BytesIO(file.read())
                    metrics = clf_train(df_stream, model_name, app.config['MODEL_DIR'])
                    results = {
                        "model_name": model_name,
                        "accuracy": metrics['accuracy'],
                        "loss": metrics['loss'],
                        "data_count": metrics['data_count']
                    }
                except Exception as e:
                    error = str(e)

        # === PREDIKSI MANUAL ===
        elif action == 'predict_manual':
            model_name = request.form.get('model_select')
            if not model_name:
                error = "Pilih model terlebih dahulu!"
            else:
                try:
                    values = []
                    for col in TARGET_COLUMNS:
                        val = request.form.get(col)
                        if val is None or val.strip() == '':
                            raise ValueError(f"Nilai {col} tidak boleh kosong!")
                        values.append(float(val))
                    
                    prediction = predict_manual(values, model_name, app.config['MODEL_DIR'])
                except Exception as e:
                    error = str(e)

    return render_template('klasifikasi.html',
                           models=clf_models,
                           results=results,
                           error=error,
                           prediction=prediction,
                           default_values=default_values,
                           target_columns=TARGET_COLUMNS)

if __name__ == '__main__':
    print("Akses: http://127.0.0.1:5000")
    app.run(debug=True)
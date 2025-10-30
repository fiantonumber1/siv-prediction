# app.py
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
import io
from model import preprocess_data, train_and_save, predict_with_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_DIR'] = 'models'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
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
                   # Di dalam action == 'train'
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
            steps = int(request.form.get('steps', 1))  # ← ambil dari input HTML
            if not file or not model_name:
                error = "File dan model wajib dipilih!"
            else:
                try:
                    df_stream = io.StringIO(file.stream.read().decode('utf-8'))
                    prediction, plot_html = predict_with_model(df_stream, model_name, app.config['MODEL_DIR'], steps=steps)
                except Exception as e:
                    error = str(e)


    return render_template('index.html', 
                         models=models, 
                         results=results, 
                         error=error, 
                         plot_html=plot_html, 
                         prediction=prediction)

if __name__ == '__main__':
    print("Akses: http://127.0.0.1:5000")
    app.run(debug=True)
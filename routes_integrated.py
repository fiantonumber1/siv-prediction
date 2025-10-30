# routes_integrated.py
from flask import Blueprint, request, render_template, current_app
import os
from integrated_model import predict_integrated

integrated_bp = Blueprint('integrated', __name__, template_folder='templates')

@integrated_bp.route('/integrated', methods=['GET', 'POST'])
def integrated():
    model_dir = current_app.config['MODEL_DIR']
    
    # Daftar model
    forecast_models = [f.split('_scaler')[0] for f in os.listdir(model_dir) if f.endswith('_scaler.pkl')]
    clf_models = [f.split('_clf')[0] for f in os.listdir(model_dir) if f.endswith('_clf.h5')]

    error = None
    result = None

    if request.method == 'POST':
        file = request.files.get('file')
        forecast_model = request.form.get('forecast_model')
        clf_model = request.form.get('clf_model')
        steps = int(request.form.get('steps', 1))

        if not file or not forecast_model or not clf_model:
            error = "File, model forecast, dan model klasifikasi wajib dipilih!"
        else:
            try:
                file_content = file.read()
                result = predict_integrated(file_content, forecast_model, clf_model, model_dir, steps)
            except Exception as e:
                error = str(e)

    return render_template(
        'integrated.html',
        forecast_models=forecast_models,
        clf_models=clf_models,
        result=result,
        error=error
    )
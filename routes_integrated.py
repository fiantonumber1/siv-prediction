# routes_integrated.py
from flask import Blueprint, request, render_template, current_app
import os
from integrated_model import predict_integrated
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

integrated_bp = Blueprint('integrated', __name__, template_folder='templates')

@integrated_bp.route('/integrated', methods=['GET', 'POST'])
def integrated():
    model_dir = current_app.config['MODEL_DIR']
    
    forecast_models = [f.split('_scaler')[0] for f in os.listdir(model_dir) if f.endswith('_scaler.pkl')]
    clf_models = [f.split('_clf')[0] for f in os.listdir(model_dir) if f.endswith('_clf.h5')]

    error = None
    result = None
    fault_plot_html = None

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

                # --- BUAT PLOT KLASIFIKASI ---
                # routes_integrated.py (bagian plot klasifikasi – UPDATE INI)
                # --- BUAT PLOT KLASIFIKASI ---
                steps_list = [f"Step {i+1}" for i in range(steps)]
                fault_labels = [
                    'BC Flt', 'InConv Flt', 'Inv Flt',
                    'UX BC Flt', 'UX InConv Flt', 'UX Inv Flt'
                ]

                fig = make_subplots(
                    rows=steps, cols=1,
                    subplot_titles=steps_list,
                    vertical_spacing=0.12,
                    shared_xaxes=True
                )

                # WARNA TAJAM: MERAH = FAULT (1), HIJAU = NORMAL (0)
                color_map = {0: '#27ae60', 1: '#e74c3c'}  # hijau, merah

                for idx, step_key in enumerate(steps_list):
                    clf = result['classification_per_step'][step_key]
                    values = [clf['binary'][label] for label in [
                        'SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres',
                        'Ux_SIV_MajorBCFltPres', 'Ux_SIV_MajorInputConvFltPres', 'Ux_SIV_MajorInverterFltPres'
                    ]]
                    probs = [clf['probabilities'][label] for label in [
                        'SIV_MajorBCFltPres', 'SIV_MajorInputConvFltPres', 'SIV_MajorInverterFltPres',
                        'Ux_SIV_MajorBCFltPres', 'Ux_SIV_MajorInputConvFltPres', 'Ux_SIV_MajorInverterFltPres'
                    ]]

                    fig.add_trace(
                        go.Bar(
                            x=fault_labels,
                            y=values,
                            marker_color=[color_map[v] for v in values],  # MERAH jika 1, HIJAU jika 0
                            text=[f"{p:.1%}" if v == 1 else "" for v, p in zip(values, probs)],  # hanya tampilkan % jika fault
                            textposition='outside',
                            textfont=dict(color='white', size=10),
                            hovertemplate=
                                "<b>%{x}</b><br>" +
                                "Status: <b>%{y}</b><br>" +
                                "Probabilitas: <b>%{text}</b><extra></extra>",
                            name=step_key
                        ),
                        row=idx + 1, col=1
                    )

                    # Tambahkan garis horizontal di 0.5 (threshold)
                    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", row=idx + 1, col=1)

                fig.update_layout(
                    height=220 * steps,
                    showlegend=False,
                    title_text="Klasifikasi Fault per Langkah Prediksi",
                    barmode='stack',
                    font=dict(family="Inter, sans-serif")
                )
                fig.update_yaxes(range=[0, 1.4], tick0=0, dtick=1, showticklabels=False, showgrid=False)
                fig.update_xaxes(tickangle=0)

                fault_plot_html = fig.to_html(full_html=False, include_plotlyjs=False)

            except Exception as e:
                error = str(e)

    return render_template(
        'integrated.html',
        forecast_models=forecast_models,
        clf_models=clf_models,
        result=result,
        error=error,
        fault_plot_html=fault_plot_html
    )
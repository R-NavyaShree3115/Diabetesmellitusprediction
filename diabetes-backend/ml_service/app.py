# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF
from io import BytesIO
from collections import Counter

# ------------------------------
# Flask Config
# ------------------------------
app = Flask(__name__)
app.secret_key = 'securekey123'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------------------
# Load Models & Scaler
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

models = {
    'logistic': joblib.load(os.path.join(MODEL_PATH, "logistic_model.pkl")),
    'decision_tree': joblib.load(os.path.join(MODEL_PATH, "decision_tree_model.pkl")),
    'random_forest': joblib.load(os.path.join(MODEL_PATH, "random_forest_model.pkl")),
    'svm': joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl")),
    'knn': joblib.load(os.path.join(MODEL_PATH, "knn_model.pkl"))
}

ensemble_model = joblib.load(os.path.join(MODEL_PATH, "ensemble_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))

REQUIRED_FEATURES = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI','DiabetesPedigreeFunction','Age'
]

# ------------------------------
# Static login
# ------------------------------
VALID_USERNAME = 'admin'
VALID_PASSWORD = 'admin'

# ------------------------------
# Helper function
# ------------------------------
def classify_risk(predictions, features_array):
    majority_vote = Counter(predictions).most_common(1)[0][0]
    glucose = features_array[0][1]
    bmi = features_array[0][5]

    if majority_vote == 0:
        if 100 <= glucose <= 125 or 25 <= bmi <= 29.9:
            return "Prediabetes"
        else:
            return "No Diabetes"
    else:
        return "Diabetes"

# ------------------------------
# Routes
# ------------------------------
df_preview = None

@app.route('/')
def index():
    return render_template('loginpage.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        flash('✅ Login successful!', 'success')
        return redirect(url_for('upload'))
    else:
        flash('❌ Invalid credentials', 'danger')
        return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global df_preview
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('⚠️ No file selected', 'danger')
            return redirect(request.url)
        if not file.filename.endswith('.csv'):
            flash('⚠️ Upload a CSV file', 'danger')
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        try:
            df_preview = pd.read_csv(filepath)
            flash('✅ File uploaded!', 'success')
            return render_template('upload.html', tables=[df_preview.head().to_html(classes='data', header="true")])
        except Exception as e:
            flash(f'⚠️ Error reading CSV: {e}', 'danger')
            return redirect(request.url)
    return render_template('upload.html', tables=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f]) for f in REQUIRED_FEATURES]
        features_array = np.array(features).reshape(1, -1)
        features_df = pd.DataFrame(features_array, columns=REQUIRED_FEATURES)

        scaled_features = scaler.transform(features_df)

        predictions = [
            models['logistic'].predict(scaled_features)[0],
            models['decision_tree'].predict(features_df)[0],
            models['random_forest'].predict(features_df)[0],
            models['svm'].predict(scaled_features)[0],
            models['knn'].predict(scaled_features)[0]
        ]

        prediction_label = classify_risk(predictions, features_array)
        return render_template('premium.html', result={'Prediction': prediction_label})
    except Exception as e:
        flash(f"⚠️ Prediction error: {e}", 'danger')
        return redirect(url_for('upload'))

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.form.to_dict()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(0, 10, "Smart Diabetes Prediction Report", ln=True, align='C')
        pdf.ln(10)

        for key, val in data.items():
            pdf.cell(0, 10, f"{key}: {val}", ln=True)

        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)

        return send_file(
            pdf_output,
            as_attachment=True,
            download_name="prediction_report.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        flash(f"⚠️ PDF error: {e}", 'danger')
        return redirect(url_for('upload'))

@app.route('/test')
def test():
    return jsonify({"message": "✅ Server running!"})

# ------------------------------
# Run Flask App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)

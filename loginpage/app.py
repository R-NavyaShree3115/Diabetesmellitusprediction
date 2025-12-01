# app.py
import json
import os
from flask import (
    Flask, render_template, request, jsonify, send_from_directory,
    redirect, session, url_for, send_file
)
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from flask_cors import CORS   # type: ignore
from flask_sqlalchemy import SQLAlchemy  # type: ignore
from flask_bcrypt import Bcrypt  # type: ignore
from datetime import timedelta, datetime
from sqlalchemy.engine import Engine  # type: ignore
from sqlalchemy import event, text  # type: ignore
from sqlalchemy.schema import DropTable  # type: ignore
from sqlalchemy.ext.compiler import compiles  # type: ignore
from io import BytesIO
from openai import OpenAI  # type: ignore
from typing import Dict, Any
import pandas as pd
import numpy as np
from xgboost import XGBClassifier # type: ignore


# Optional PDF generator
try:
    from fpdf import FPDF  # type: ignore
    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False

# ------------------- Flask App Setup -------------------
app = Flask(__name__)
app.secret_key = os.environ.get("APP_SECRET", os.urandom(24))
app.config['HOME'] = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['HOME'], 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
MODEL_FOLDER = os.path.join(app.config['HOME'], 'models')
os.makedirs(MODEL_FOLDER, exist_ok=True)
CORS(app)

# ------------------- Globals -------------------
model = None            # diabetes prediction model (ensemble)
scaler = None
healthplan_model = None
diet_enc = None
exercise_enc = None
yoga_enc = None

# ------------------- File Paths -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# ------------------- Extensions / DB -------------------
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:yourpassword@localhost/diabetes_db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ------------------- SQL Query Timing Logging -------------------
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = datetime.now()

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = datetime.now() - context._query_start_time
    app.logger.debug(f"SQL Execution Time: {total}")


# -------------------- DATABASE MODELS --------------------
class Patient(db.Model):
    __tablename__ = 'patient'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

    # Pima features
    pregnancies = db.Column(db.Integer, nullable=True, default=0)
    glucose = db.Column(db.Float, nullable=True, default=100.0)
    blood_pressure = db.Column(db.Float, nullable=True, default=70.0)
    skin_thickness = db.Column(db.Float, nullable=True, default=20.0)
    insulin = db.Column(db.Float, nullable=True, default=0.0)
    bmi = db.Column(db.Float, nullable=True, default=25.0)
    diabetes_pedigree_function = db.Column(db.Float, nullable=True, default=0.5)
    age = db.Column(db.Integer, nullable=True, default=30)
    last_diabetes_outcome = db.Column(db.Integer, nullable=True, default=0)

class Doctor(db.Model):
    __tablename__ = 'doctor'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class HealthPlan(db.Model):
    __tablename__ = 'health_plan'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False, unique=True)
    diet_plan = db.Column(db.Text)
    exercise_plan = db.Column(db.Text)
    yoga_plan = db.Column(db.Text)

class GlucoseTracking(db.Model):
    __tablename__ = 'glucose_tracking'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    glucose = db.Column(db.Float, nullable=True)
    insulin = db.Column(db.Float, nullable=True)
    dpf = db.Column('dpf', db.Float, nullable=True)
    age = db.Column(db.Integer, nullable=True)
    outcome = db.Column(db.Integer, nullable=True)
    reading_timestamp = db.Column('reading_timestamp', db.DateTime, default=datetime.now)

class BpTracking(db.Model):
    __tablename__ = 'bp_tracking'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    bp_value = db.Column(db.Float, nullable=False)
    reading_timestamp = db.Column('reading_timestamp', db.DateTime, default=datetime.now)

# Allow dropping tables in MySQL dev environment if needed (keeps safe for sqlite too)
@compiles(DropTable, "mysql")
def _compile_drop_table(element, compiler, **kwargs):
    return "SET FOREIGN_KEY_CHECKS=0; DROP TABLE IF EXISTS %s;" % compiler.process(element.element, **kwargs)

@compiles(DropTable, "sqlite")
def _compile_drop_table_sqlite(element, compiler, **kwargs):
    return "DROP TABLE IF EXISTS %s;" % compiler.process(element.element, **kwargs)

with app.app_context():
    try:
        db.create_all()
        app.logger.info("✅ Database tables ensured.")
    except Exception as e:
        app.logger.error(f"DB create_all error: {e}")

# ------------------- Helper Function: Authentication Decorator -------------------
def login_required(role=None):
    def wrapper(f):
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('home', message='Login required'))
            if role and session.get('role') != role:
                return jsonify({'error': 'Unauthorized role'}), 403
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function
    return wrapper

# ------------------- Page Routes -------------------
@app.route('/')
def home():
    session.clear() 
    if 'user_id' in session:
        if session.get('role') == 'patient':
            return redirect(url_for('uploadpage'))
        elif session.get('role') == 'doctor':
            return redirect(url_for('doctor_dashboard'))
    return render_template('loginpage.html')

@app.route('/images/<path:filename>')
def serve_images(filename):
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    return send_from_directory(image_dir, filename)

@app.route('/upload')
@login_required(role='patient')
def uploadpage():
    return render_template('uploadpage.html')

@app.route('/premium')
@login_required(role='patient')
def premium():
    return render_template('premium.html')

@app.route('/doctor_dashboard')
@login_required(role='doctor')
def doctor_dashboard():
    return render_template('doctor_dashboard.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# -------------------------- AUTH ROUTES --------------------------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    role = data.get('type')  # 'patient' or 'doctor'

    if not email or not password or not role:
        return jsonify({'error': 'Missing email, password, or role'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    try:
        if role == 'patient':
            if Patient.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already registered as patient'}), 409
            new_user = Patient(email=email, password=hashed_password)
        elif role == 'doctor':
            if Doctor.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already registered as doctor'}), 409
            new_user = Doctor(email=email, password=hashed_password)
        else:
            return jsonify({'error': 'Invalid role type'}), 400

        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': f'{role.capitalize()} registered successfully!'}), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Database error during signup: {e}")
        return jsonify({'error': 'A database error occurred during signup.'}), 500
    

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    role = data.get('type')

    if not email or not password or not role:
        return jsonify({'error': 'Missing email, password, or role'}), 400

    user = None
    redirect_url = url_for('home')

    if role == 'patient':
        user = Patient.query.filter_by(email=email).first()
        redirect_url = url_for('uploadpage')
    elif role == 'doctor':
        user = Doctor.query.filter_by(email=email).first()
        redirect_url = url_for('doctor_dashboard')
    else:
        return jsonify({'error': 'Invalid role type'}), 400

    if user and bcrypt.check_password_hash(user.password, password):
        session.permanent = True
        session['user_id'] = user.id
        session['role'] = role
        session['email'] = user.email
        return jsonify({'success': 'Login successful', 'redirect': redirect_url}), 200
    elif user is None:
        return jsonify({'error': 'User not found, sign in'}), 404
    else:
        return jsonify({'error': 'Incorrect password or email'}), 401


# ------------------- CSV Upload & Train Diabetes Model -------------------
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'dataset' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded!'}), 400

    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected!'}), 400

    # Save file
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(path)

    # Read CSV
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to read CSV: {e}'}), 400

    # Validate PIMA columns
    pima_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    if not all(col in df.columns for col in pima_cols) or 'Outcome' not in df.columns:
        return jsonify({
            'status': 'error',
            'message': 'CSV must contain all PIMA features and the "Outcome" column!'
        }), 400

    if df['Outcome'].nunique() < 2:
        return jsonify({
            'status': 'error',
            'message': 'CSV must contain "Outcome" column with >=2 classes!'
        }), 400

    preview_html = df.head(10).to_html(classes='table table-striped', index=False)

    # ------------------- Train Diabetes Models -------------------
    X = df[pima_cols]
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    global scaler, model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Individual classifiers
    logistic_model = LogisticRegression(max_iter=500)
    logistic_model.fit(X_train_scaled, y_train)

    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)

    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)

    # Ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', logistic_model),
            ('dt', decision_tree_model),
            ('rf', random_forest_model),
            ('svm', svm_model),
            ('knn', knn_model)
        ],
        voting='hard'
    )
    ensemble_model.fit(X_train_scaled, y_train)

    # ------------------- Save Models & Scaler -------------------
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    joblib.dump(ensemble_model, os.path.join(MODEL_FOLDER, 'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_FOLDER, 'scaler.pkl'))

    # Update global model
    model = ensemble_model

    # ------------------- Compute Accuracies -------------------
    accuracies = {
        'Logistic Regression': round(logistic_model.score(X_test_scaled, y_test) * 100, 2),
        'Decision Tree': round(decision_tree_model.score(X_test, y_test) * 100, 2),
        'Random Forest': round(random_forest_model.score(X_test, y_test) * 100, 2),
        'SVM': round(svm_model.score(X_test_scaled, y_test) * 100, 2),
        'KNN': round(knn_model.score(X_test_scaled, y_test) * 100, 2),
        'Ensemble': round(ensemble_model.score(X_test_scaled, y_test) * 100, 2)
    }

    best_model = max(accuracies, key=accuracies.get)

    return jsonify({
        'status': 'success',
        'preview': preview_html,
        'accuracies': accuracies,
        'best_model': best_model,
        'message': 'Models trained successfully!'
    })


# ------------------- Predict Diabetes Risk + HealthPlan -------------------
@app.route('/predict', methods=['POST'])
@login_required(role='patient')
def predict():
    try:
        data = request.get_json() or request.form.to_dict()
        patient_id = session.get('user_id')
        if not patient_id:
            return jsonify({"status": "error", "message": "User session expired"}), 401

        required_fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing field: {field}"}), 400
            data[field] = float(data[field])

        # Prediction
        features_df = pd.DataFrame([data], columns=required_fields)
        global model, scaler
        if not model or not scaler:
            return jsonify({"status": "error", "message": "Model not loaded"}), 503

        X_scaled = scaler.transform(features_df)
        prediction = int(model.predict(X_scaled)[0])
        prediction_label = "High" if prediction == 1 else "Low"

        # Advice
        advice = ("⚠️ High risk of diabetes. Consult doctor." if prediction == 1
                  else "✅ Low risk. Maintain healthy habits.")

        # Update patient
        patient = db.session.get(Patient, patient_id)
        for k, col in {
            "Pregnancies":"pregnancies", "Glucose":"glucose", "BloodPressure":"blood_pressure",
            "SkinThickness":"skin_thickness", "Insulin":"insulin", "BMI":"bmi",
            "DiabetesPedigreeFunction":"diabetes_pedigree_function", "Age":"age"
        }.items():
            setattr(patient, col, data[k])
        patient.last_diabetes_outcome = prediction

        # Log glucose and BP
        db.session.add(GlucoseTracking(patient_id=patient.id, glucose=data["Glucose"],
                                       insulin=data["Insulin"],
                                       dpf=data["DiabetesPedigreeFunction"],
                                       age=int(data["Age"]),
                                       outcome=prediction))
        db.session.add(BpTracking(patient_id=patient.id, bp_value=data["BloodPressure"]))

        db.session.commit()

        return jsonify({
            "status":"success",
            "diabetes_risk": prediction_label,
            "Outcome": prediction,
            "advice": advice,
            
            "message":"Prediction and logs updated."
        }), 200

    except Exception as e:
        app.logger.error(f"/predict ERROR: {e}")
        db.session.rollback()
        return jsonify({"status":"error","message":"Internal server error"}), 500



# ------------------- Dashboard -------------------
@app.route('/dashboard')
@login_required(role='patient')
def dashboard():
    patient_id = session.get('user_id')
    patient = db.session.get(Patient, patient_id)

    if not patient:
        return redirect(url_for('home'))

    # Load health plan directly from DB (INSTANT)
    hp = HealthPlan.query.filter_by(patient_id=patient_id).first()

    def safe_load(x):
        try:
            return json.loads(x) if x else []
        except:
            return []

    diet_recommendations = safe_load(hp.diet_plan) if hp else []
    exercise_recommendations = safe_load(hp.exercise_plan) if hp else []
    yoga_recommendations = (safe_load(hp.yoga_plan)[0] if hp and hp.yoga_plan else None)

    # ---- Glucose chart ----
    glucose_records = (
        GlucoseTracking.query.filter_by(patient_id=patient_id)
        .order_by(GlucoseTracking.reading_timestamp.desc())
        .limit(30)
        .all()
    )
    glucose_records.reverse()

    glucose_dates = [
        r.reading_timestamp.strftime('%Y-%m-%d %H:%M') if r.reading_timestamp else ""
        for r in glucose_records
    ]
    glucose_values = [float(r.glucose or 0) for r in glucose_records]

    # ---- BP chart ----
    bp_records = (
        BpTracking.query.filter_by(patient_id=patient_id)
        .order_by(BpTracking.reading_timestamp.desc())
        .limit(30)
        .all()
    )
    bp_records.reverse()

    bp_dates = [
        r.reading_timestamp.strftime('%Y-%m-%d %H:%M') if r.reading_timestamp else ""
        for r in bp_records
    ]
    bp_values = [float(r.bp_value or 0) for r in bp_records]

    # ---- PIMA context ----
    pima_context = {
        "Pregnancies": float(patient.pregnancies or 0),
        "Glucose": float(patient.glucose or 0),
        "BloodPressure": float(getattr(patient, 'blood_pressure', 0)),
        "SkinThickness": float(patient.skin_thickness or 0),
        "Insulin": float(patient.insulin or 0),
        "BMI": float(patient.bmi or 0),
        "DiabetesPedigreeFunction": float(patient.diabetes_pedigree_function or 0),
        "Age": float(patient.age or 0),
    }

    last_outcome = int(patient.last_diabetes_outcome or 0)

    return render_template(
        "dashboard.html",
        patient={"id": patient.id, "name": f"P{patient.id} (Age {patient.age})"},
        dates=glucose_dates,
        glucose_values=glucose_values,
        bp_dates=bp_dates,
        bp_values=bp_values,
        diet_recommendations=diet_recommendations,
        exercise_recommendations=exercise_recommendations,
        yoga_recommendations=yoga_recommendations,
        last_outcome=last_outcome,
        pima_context=pima_context
    )

import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier # type: ignore

# ------------ Load CSV Path ------------
csv_path = os.path.join(os.path.dirname(__file__), "data", "diet.csv")

# ------------ Global Cache ------------
health_model = None
health_df = None


def train_healthplan_model():
    """Train XGBoost on diet.csv ONLY ONCE using glucose, BMI, PedigreeFunction, age."""

    global health_model, health_df

    # If already trained → reuse (FAST)
    if health_model is not None and health_df is not None:
        return health_model, health_df

    df = pd.read_csv(csv_path)

    # Input features
    X = df[['glucose', 'BMI', 'PedigreeFunction', 'age']]

    # Target label = row index
    df['label'] = np.arange(len(df))
    y = df['label']

    # Train model
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss"
    )
    model.fit(X, y)

    # Save in cache
    health_model = model
    health_df = df

    return model, df



# ---------------------- HEALTH PLAN PAGE ----------------------
@app.route('/healthplan')
@login_required(role='patient')
def healthplan():
    try:
        patient_id = session.get('user_id')
        patient = db.session.get(Patient, patient_id)

        if not patient:
            return redirect(url_for('home'))

        model, df = train_healthplan_model()
        from scipy.spatial.distance import cdist

        # Patient input
        inp = np.array([[float(patient.glucose or 0),
                        float(patient.bmi or 0),
                        float(patient.diabetes_pedigree_function or 0),
                        float(patient.age or 0)]])

        # Compute distance from input to each row in health_df
        feature_cols = ['glucose', 'BMI', 'PedigreeFunction', 'age']
        distances = cdist(inp, df[feature_cols].values, metric='euclidean')[0]

        # Find the closest row
        closest_idx = np.argmin(distances)
        row = df.iloc[closest_idx]

        plan = {
            "diet": row['Diet_plan'].split(",") if isinstance(row['Diet_plan'], str) else [],
            "exercise": row['Exercise_plan'].split(";") if isinstance(row['Exercise_plan'], str) else [],
            "yoga": row['Yoga_Recommendation'],
        }

        # ---------- SAVE INTO DB ----------
        hp = HealthPlan.query.filter_by(patient_id=patient_id).first()
        if not hp:
            hp = HealthPlan(patient_id=patient_id)

        hp.diet_plan     = json.dumps(plan["diet"])
        hp.exercise_plan = json.dumps(plan["exercise"])
        hp.yoga_plan     = json.dumps([plan["yoga"]])  # store list

        db.session.add(hp)
        db.session.commit()
        # ----------------------------------

        return render_template(
            "healthplan.html",
            diet_recommendations=plan["diet"],
            exercise_recommendations=plan["exercise"],
            yoga_recommendations=plan["yoga"]
        )

    except Exception as e:
        print("HealthPlan ERROR:", e)
        return render_template(
            "healthplan.html",
            diet_recommendations=[],
            exercise_recommendations=[],
            yoga_recommendations=None
        )
from flask import Flask, render_template, request, jsonify
import os
from openai import OpenAI # type: ignore
# ------------------- DOCTOR CHAT WITH OPENAI -------------------
app.secret_key = "supersecret"
# ------------------- OPENAI CLIENT -------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")
client = OpenAI(api_key=OPENAI_API_KEY)

@app.route("/doctorconnect")
def doctorconnect():
    return render_template("doctorconnect.html")

@app.route("/doctor_chat", methods=["POST"])
@login_required(role="patient")
def doctor_chat():
    data = request.get_json()
    user_msg = data.get("message", "").strip()

    if not user_msg:
        return jsonify({"reply": "Please enter a message."})

    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=f"You are a doctor. Patient says: {user_msg}. "
                  "Respond concisely, medically accurate, in simple terms."
        )
        # Extract text correctly
        doctor_reply = response.output[0].content[0].text
        return jsonify({"reply": doctor_reply})

    except Exception as e:
        app.logger.error(f"Doctor Chat ERROR: {e}")
        return jsonify({
            "reply": "⚠️ Sorry, the AI service is currently unavailable. Please try again later."
        })



from fpdf import FPDF # type: ignore
from flask import send_file
import io
import json

@app.route("/download_report")
@login_required(role="patient")
def download_report():
    patient_id = session.get('user_id')
    patient = db.session.get(Patient, patient_id)
    if not patient:
        return redirect(url_for('home'))

    # Fetch health plan
    hp = HealthPlan.query.filter_by(patient_id=patient_id).first()
    diet = json.loads(hp.diet_plan) if hp and hp.diet_plan else []
    exercise = json.loads(hp.exercise_plan) if hp and hp.exercise_plan else []
    yoga = json.loads(hp.yoga_plan)[0] if hp and hp.yoga_plan else "N/A"

    # Fetch last 10 readings
    glucose_records = GlucoseTracking.query.filter_by(patient_id=patient_id).order_by(GlucoseTracking.reading_timestamp.desc()).limit(10).all()
    bp_records = BpTracking.query.filter_by(patient_id=patient_id).order_by(BpTracking.reading_timestamp.desc()).limit(10).all()

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CareConnect - Health Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0, 10, f"Patient ID: {patient.id}", ln=True)
    pdf.cell(0, 10, f"Age: {patient.age}", ln=True)
    pdf.cell(0, 10, f"Last Diabetes Outcome: {'High' if patient.last_diabetes_outcome==1 else 'Low'}", ln=True)
    pdf.ln(5)

    # Health Plan
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Health Plan:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Diet:", ln=True)
    for item in diet:
        pdf.cell(0, 8, f"- {item}", ln=True)
    pdf.cell(0, 10, "Exercise:", ln=True)
    for item in exercise:
        pdf.cell(0, 8, f"- {item}", ln=True)
    pdf.cell(0, 10, f"Yoga: {yoga}", ln=True)
    pdf.ln(5)

    # Glucose readings
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Recent Glucose Readings (mg/dL):", ln=True)
    pdf.set_font("Arial", "", 12)
    for r in reversed(glucose_records):
        pdf.cell(0, 8, f"{r.reading_timestamp.strftime('%Y-%m-%d %H:%M')} -> {r.glucose}", ln=True)

    # BP readings
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Recent BP Readings (mmHg):", ln=True)
    pdf.set_font("Arial", "", 12)
    for r in reversed(bp_records):
        pdf.cell(0, 8, f"{r.reading_timestamp.strftime('%Y-%m-%d %H:%M')} -> {r.bp_value}", ln=True)

    # Output PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # returns string, encode to bytes
    pdf_buffer = io.BytesIO(pdf_bytes)

    # Send as attachment
    return send_file(pdf_buffer, as_attachment=True, download_name="health_report.pdf", mimetype="application/pdf")

# ------------------- Run App -------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run( port=int(os.environ.get('PORT', 5000)), debug=True)   
import json
import os
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, session, url_for
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from flask_cors import CORS # type: ignore
from flask_sqlalchemy import SQLAlchemy # type: ignore
from flask_bcrypt import Bcrypt # type: ignore
from datetime import timedelta, datetime 
from sqlalchemy.engine import Engine # type: ignore
from sqlalchemy import event # type: ignore
from sqlalchemy.schema import DropTable # type: ignore
from sqlalchemy.ext.compiler import compiles # type: ignore
from sqlalchemy import text # type: ignore

# ------------------- Flask App Setup -------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['HOME'] = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)
CORS(app)

app.permanent_session_lifetime = timedelta(days=1)

# ------------------- Globals -------------------
model = None
scaler = None
healthplan_model = None
diet_enc = None
exercise_enc = None
yoga_enc = None

# ------------------- File Paths -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# ------------------- Extensions -------------------
# MySQL Configuration using SQLAlchemy
# NOTE: Replace 'yourpassword' with your actual MySQL password
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:yourpassword@localhost/diabetes_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Ensure connection pooling settings for MySQL
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._query_start_time = datetime.now()

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = datetime.now() - context._query_start_time
    print(f"SQL Execution Time: {total}")

# ------------------- Helper Function: Authentication Decorator -------------------
def login_required(role=None):
    """Decorator to check if user is logged in and optionally checks their role."""
    def wrapper(f):
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('home', message='Login required'))
            if role and session.get('role') != role:
                return jsonify({'error': 'Unauthorized role'}), 403
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__ # Fix for flask routing 
        return decorated_function
    return wrapper

# -------------------- DATABASE MODELS --------------------
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    
    # Pima features (Last known summary state, matching SQL patient table)
    pregnancies = db.Column(db.Integer, nullable=True)
    glucose = db.Column(db.Integer, nullable=True)
    blood_pressure = db.Column(db.Integer, nullable=True)
    skin_thickness = db.Column(db.Integer, nullable=True)
    insulin = db.Column(db.Integer, nullable=True)
    bmi = db.Column(db.Float, nullable=True)
    diabetes_pedigree_function = db.Column(db.Float, nullable=True)
    age = db.Column(db.Integer, nullable=True)
    last_diabetes_outcome = db.Column(db.Integer, nullable=True)
    
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    
class HealthPlan(db.Model):
    __tablename__ = 'health_plan' # lowercase table name is safer for MySQL
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False, unique=True)
    diet_plan = db.Column(db.Text)
    exercise_plan = db.Column(db.Text)
    yoga_plan = db.Column(db.Text) 

# --- UPDATED MODEL: Consolidated Glucose Tracking and Prediction History ---
class GlucoseTracking(db.Model):
    """
    Model to store the full PIMA feature set and prediction outcome 
    for every submission, timestamped for tracking.
    """
    __tablename__ = 'glucose_tracking'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    
    # Pima Features (inputs)
    pregnancies = db.Column(db.Integer, nullable=True)
    glucose = db.Column(db.Float, nullable=True)
    blood_pressure = db.Column(db.Float, nullable=True)
    skin_thickness = db.Column(db.Float, nullable=True)
    insulin = db.Column(db.Float, nullable=True)
    bmi = db.Column(db.Float, nullable=True)
    
    # Map 'DiabetesPedigreeFunction' from model to 'dpf' in SQL table
    # This aligns the model with the SQL query structure from your trace
    dpf = db.Column('dpf', db.Float, nullable=True) 
    
    age = db.Column(db.Integer, nullable=True)
    
    # Prediction Outcome (output)
    outcome = db.Column(db.Integer, nullable=True) 
    
    # Timestamp column
    reading_timestamp = db.Column('reading_timestamp', db.DateTime, default=datetime.now)


# --- FIX: Ensure SQLAlchemy can drop tables with foreign key constraints ---
# This is a common pattern to handle tables that might still exist from old schema versions
@compiles(DropTable, "mysql")
def _compile_drop_table(element, compiler, **kwargs):
    """
    Adds CASCADE/IF EXISTS and foreign key check bypass for MySQL DROP TABLE commands.
    This ensures dependent tables/old tables can be cleanly dropped in development.
    """
    return "SET FOREIGN_KEY_CHECKS=0; DROP TABLE IF EXISTS %s;" % compiler.process(element.element, **kwargs)

@compiles(DropTable, "sqlite")
def _compile_drop_table_sqlite(element, compiler, **kwargs):
    """SQLite equivalent (though the error is MySQL-specific)"""
    return "DROP TABLE IF EXISTS %s;" % compiler.process(element.element, **kwargs)


with app.app_context():
    # 🛑 FIX FOR OperationalError: Cannot drop table 'patient'
    # We use db.metadata.drop_all(db.engine) which is generally more robust
    # than db.drop_all() alone at handling foreign key ordering conflicts.
    # Additionally, we manually ensure old schema tables are removed using raw SQL text.

    try:
        # Use raw SQL to explicitly drop the old table name mentioned in the error
        # and temporarily disable FK checks for a clean drop sequence.
        db.session.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        db.session.execute(text("DROP TABLE IF EXISTS prediction_history"))
        db.session.commit()

        # Drop all tables in dependency order (should work after the above cleanup)
        db.metadata.drop_all(db.engine)
        
        # Re-enable checks and recreate all tables.
        db.session.execute(text("SET FOREIGN_KEY_CHECKS=1"))
        db.session.commit()

        db.create_all()
        print("✅ Database tables dropped and recreated successfully to apply schema updates.")

    except Exception as e:
        print(f"⚠️ Warning: Database cleanup encountered an error: {e}. Attempting to proceed with create_all().")
        db.create_all()
        db.session.execute(text("SET FOREIGN_KEY_CHECKS=1")) # Ensure checks are re-enabled
        db.session.commit()
        print("✅ Database schema creation completed (possibly retaining old data/tables).")


# ------------------- Load ML Models -------------------
diabetes_model_path = os.path.join(MODEL_FOLDER,'diabetes_model.pkl')
scaler_path = os.path.join(MODEL_FOLDER,'scaler.pkl')
healthplan_model_path = os.path.join(MODEL_FOLDER,'healthplan_model.pkl')
diet_enc_path = os.path.join(MODEL_FOLDER,'diet_encoder.pkl')
exercise_enc_path = os.path.join(MODEL_FOLDER,'exercise_encoder.pkl')
yoga_enc_path = os.path.join(MODEL_FOLDER,'yoga_encoder.pkl')

if os.path.exists(diabetes_model_path):
    diabetes_model = joblib.load(diabetes_model_path)
    scaler = joblib.load(scaler_path)
else:
    diabetes_model = None
    scaler = None

if os.path.exists(healthplan_model_path):
    healthplan_model = joblib.load(healthplan_model_path)
    diet_enc = joblib.load(diet_enc_path)
    exercise_enc = joblib.load(yoga_enc_path)
    yoga_enc = joblib.load(yoga_enc_path)
else:
    healthplan_model = diet_enc = exercise_enc = yoga_enc = None

# ------------------- Page Routes -------------------

@app.route('/')
def home():
    """Default route → Login page (or dashboard if already logged in)"""
    if 'user_id' in session:
        if session.get('role') == 'patient':
            return redirect(url_for('uploadpage')) 
        elif session.get('role') == 'doctor':
            return redirect(url_for('doctor_dashboard'))
    return render_template('loginpage.html') 

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serves static image files."""
    # Assuming 'images' directory exists next to app.py
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    return send_from_directory(image_dir, filename)

@app.route('/upload')
@login_required(role='patient')
def uploadpage():
    """Route for doctors to upload datasets for model training (redirected to here after patient login)."""
    return render_template('upload.html')

@app.route('/premium')
@login_required(role='patient')
def premium():
    """Prediction input page for patients."""
    return render_template('premium.html')


@app.route('/doctor_dashboard')
@login_required(role='doctor')
def doctor_dashboard():
    """Doctor dashboard (separate route for doctor view)."""
    return render_template('doctor_dashboard.html')

@app.route('/healthplan')
@login_required(role='patient')
def healthplan():
    """Renders the Health Plan page for the patient."""
    return render_template('healthplan.html')

@app.route('/progress')
@login_required(role='patient')
def progress():
    """Progress tracking page for patients."""
    return render_template('progress.html')

@app.route('/alerts')
@login_required(role='patient')
def alerts():
    """Alerts and notifications page."""
    return render_template('alerts.html')

@app.route('/doctorconnect')
@login_required(role='patient')
def doctorconnect():
    """Page for patients to connect with doctors."""
    return render_template('doctorconnect.html')

@app.route('/logout')
def logout():
    """Clear session and redirect to home (login page)."""
    session.clear()
    return redirect(url_for('home')) 

# --------------------------
# AUTHENTICATION API ROUTES
# --------------------------

@app.route('/signup', methods=['POST'])
def signup():
    """Handles patient and doctor registration."""
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    role = data.get('type') # 'patient' or 'doctor'

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
        print(f"Database error during signup: {e}")
        return jsonify({'error': 'A database error occurred during signup.'}), 500

@app.route('/login', methods=['POST'])
def login():
    """Handles user login and session creation."""
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

    # Correct use of flask_bcrypt to check the password hash
    if user and bcrypt.check_password_hash(user.password, password):
        # Successful Login: Set session data
        session.permanent = True
        session['user_id'] = user.id
        session['role'] = role
        session['email'] = user.email 
        return jsonify({'success': 'Login successful', 'redirect': redirect_url}), 200
    elif user is None:
        return jsonify({'error': 'User not found, sign in'}), 404
    else:
        return jsonify({'error': 'Incorrect password or email'}), 401
    
# ------------------- HEALTH PLAN MODEL TRAINING FUNCTION -------------------
def train_healthplan_model(df):
    """
    Trains an ML model to predict Health Plan components (Diet, Exercise, Yoga)
    based on the patient's health parameters and diabetes Outcome.
    """
    global healthplan_model, diet_enc, exercise_enc, yoga_enc
    global scaler # Uses the scaler trained for the diabetes model
    
    plan_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    plan_targets = ['Diet_Plan_Type', 'Exercise_Intensity', 'Yoga_Routine']

    # 1. Input Validation
    if scaler is None:
        print("⚠️ Scaler is not initialized. Cannot train Health Plan model. Ensure PIMA model trained first.")
        return False
        
    if not all(col in df.columns for col in plan_features + plan_targets):
        print("⚠️ Health Plan training skipped: Missing required columns (PIMA features, Outcome, or plan targets) in the dataset.")
        return False 

    X = df[plan_features]
    Y_targets = df[plan_targets].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # 2. Encode Target Variables (Categorical to Numerical)
    diet_enc = LabelEncoder()
    exercise_enc = LabelEncoder()
    yoga_enc = LabelEncoder()

    Y_targets['Diet_Plan_Type'] = diet_enc.fit_transform(Y_targets['Diet_Plan_Type'])
    Y_targets['Exercise_Intensity'] = exercise_enc.fit_transform(Y_targets['Exercise_Intensity'])
    Y_targets['Yoga_Routine'] = yoga_enc.fit_transform(Y_targets['Yoga_Routine'])
    
    Y = Y_targets.values

    # 3. Scale Features
    # Scale PIMA features (first 8 columns) and combine with 'Outcome' (9th column, unscaled)
    X_scaled_pima = scaler.transform(X.drop('Outcome', axis=1))
    X_final = np.hstack([X_scaled_pima, X['Outcome'].values.reshape(-1, 1)])

    # 4. Train Multi-Output Classifier (e.g., Random Forest)
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    healthplan_model = MultiOutputClassifier(base_clf, n_jobs=-1)
    healthplan_model.fit(X_final, Y)

    # 5. Save Model and Encoders
    joblib.dump(healthplan_model, os.path.join(MODEL_FOLDER, 'healthplan_model.pkl'))
    joblib.dump(diet_enc, os.path.join(MODEL_FOLDER, 'diet_encoder.pkl'))
    joblib.dump(exercise_enc, os.path.join(MODEL_FOLDER, 'exercise_encoder.pkl'))
    joblib.dump(yoga_enc, os.path.join(MODEL_FOLDER, 'yoga_encoder.pkl'))
    
    print("✅ Health Plan Model and Encoders trained and saved.")
    return True

# ------------------- CSV Upload & Train Models -------------------
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'dataset' not in request.files:
        return jsonify({'status':'error','message':'No file uploaded!'})

    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'status':'error','message':'No file selected!'})

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    df = pd.read_csv(path)

    if 'Outcome' not in df.columns or df['Outcome'].nunique() < 2:
        return jsonify({'status':'error','message':'CSV must contain "Outcome" column with >=2 classes!'})

    preview_html = df.head(10).to_html(classes='table table-striped', index=False)

    # --- DIABETES MODEL TRAINING ---
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models (logistic_model, decision_tree_model, random_forest_model, svm_model, knn_model, ensemble_model)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_scaled, y_train)
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest_model.fit(X_train, y_train)
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    ensemble_model = VotingClassifier(
        estimators=[('lr', logistic_model), ('dt', decision_tree_model),
                    ('rf', random_forest_model), ('svm', svm_model),
                    ('knn', knn_model)],
        voting='hard'
    )
    ensemble_model.fit(X_train_scaled, y_train)

    # Save models
    joblib.dump(ensemble_model, os.path.join(MODEL_FOLDER,'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_FOLDER,'scaler.pkl'))

    # --- HEALTH PLAN MODEL TRAINING ---
    try:
        train_healthplan_model(df)
    except Exception as e:
        print(f"⚠️ Health Plan training failed (Ignore if CSV is only PIMA data): {e}")

    # --- EVALUATION ---
    accuracies = {
        'Logistic Regression': round(logistic_model.score(X_test_scaled, y_test)*100,2),
        'Decision Tree': round(decision_tree_model.score(X_test, y_test)*100,2),
        'Random Forest': round(random_forest_model.score(X_test, y_test)*100,2),
        'SVM': round(svm_model.score(X_test_scaled, y_test)*100,2),
        'KNN': round(knn_model.score(X_test_scaled, y_test)*100,2),
        'Ensemble': round(ensemble_model.score(X_test_scaled, y_test)*100,2)
    }

    best_model = max(accuracies, key=accuracies.get)

    return jsonify({
        'status':'success',
        'preview': preview_html,
        'accuracies': accuracies,
        'best_model': best_model,
        'message':'Models trained successfully!'
    })

# Load model on startup
with app.app_context():
    if os.path.exists(os.path.join(MODEL_FOLDER,'diabetes_model.pkl')):
        model = joblib.load(os.path.join(MODEL_FOLDER,'diabetes_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_FOLDER,'scaler.pkl'))

# ------------------- Diabetes Prediction -------------------
@app.route('/predict', methods=['POST'])
@login_required(role='patient')
def predict():
    """
    Handles diabetes prediction, updates the Patient's last known state, 
    and logs the complete reading set into the GlucoseTracking history table 
    with a timestamp.
    """
    try:
        data = request.get_json()
        patient_id = session.get('user_id')

        # Feature preparation
        features = np.array([
            data['Pregnancies'], data['Glucose'], data['BloodPressure'],
            data['SkinThickness'], data['Insulin'], data['BMI'],
            data['DiabetesPedigreeFunction'], data['Age']
        ]).reshape(1, -1)

        if model is None or scaler is None:
            return jsonify({"error": "Model or scaler not initialized. Please upload data first.", "status": "failed"}), 503

        features_scaled = scaler.transform(features)
        prediction = int(model.predict(features_scaled)[0])

        advice = (
            "⚠️ High risk of diabetes detected. Consult your doctor soon."
            if prediction == 1
            else "✅ Low risk of diabetes. Maintain a healthy lifestyle."
        )

        # ✅ Update patient record
        patient = Patient.query.get(patient_id)
        if patient:
            patient.pregnancies = data['Pregnancies']
            patient.glucose = data['Glucose']
            patient.blood_pressure = data['BloodPressure']
            patient.skin_thickness = data['SkinThickness']
            patient.insulin = data['Insulin']
            patient.bmi = data['BMI']
            patient.diabetes_pedigree_function = data['DiabetesPedigreeFunction']
            patient.age = data['Age']
            patient.last_diabetes_outcome = prediction

            # ✅ Add new tracking record (preserves old readings)
            new_record = GlucoseTracking(
                patient_id=patient.id,
                pregnancies=data['Pregnancies'],
                glucose=data['Glucose'],
                blood_pressure=data['BloodPressure'],
                skin_thickness=data['SkinThickness'],
                insulin=data['Insulin'],
                bmi=data['BMI'],
                # Ensure the key here matches the column name 'dpf' in the model definition
                dpf=data['DiabetesPedigreeFunction'], 
                age=data['Age'],
                outcome=prediction
            )
            db.session.add(new_record)

        db.session.commit()

        # ✅ Also return the updated history for instant chart refresh
        history = GlucoseTracking.query.filter_by(patient_id=patient_id).order_by(
            GlucoseTracking.reading_timestamp.asc()).all()

        history_data = [{
            "timestamp": h.reading_timestamp.strftime('%Y-%m-%d %H:%M'),
            "glucose": float(h.glucose or 0)
        } for h in history]

        return jsonify({
            "prediction": prediction,
            "advice": advice,
            "status": "success",
            "history": history_data
        })

    except Exception as e:
        db.session.rollback()
        print(f"Prediction Error: {e}")
        # Return a clean error message to the frontend without exposing full traceback
        return jsonify({"error": "Failed to log prediction data. Check database schema.", "status": "failed"}), 500

# ------------------- Dashboard Route -------------------
@app.route('/dashboard')
@login_required(role='patient')
def dashboard():
    patient_id = session.get('user_id')
    patient_record = Patient.query.get(patient_id)

    # 🛡️ Safety: handle invalid/expired session
    if not patient_record:
        session.clear()
        return redirect(url_for('home'))

    # 🧠 Build patient info to show dynamically on dashboard
    patient_info = {
        'id': patient_record.id,
        'name': f"P{patient_record.id} (Age {patient_record.age or 'N/A'})"
    }

    # 📈 Fetch last 30 glucose readings (oldest → newest)
    records = (
        GlucoseTracking.query
        .filter_by(patient_id=patient_id)
        .order_by(GlucoseTracking.reading_timestamp.asc())
        .limit(30)
        .all()
    )

    # 🗓️ Convert DB data → ChartJS-friendly lists
    dates = [r.reading_timestamp.strftime('%Y-%m-%d %H:%M') for r in records]
    glucose_values = [float(r.glucose or 0) for r in records]

    # 🎯 Send data to template for graph generation
    return render_template(
        'dashboard.html',
        patient=patient_info,
        dates=dates,
        glucose_values=glucose_values
    )


# ------------------- Run App -------------------
if __name__ == '__main__':
    with app.app_context():
        pass
    app.run(debug=True)

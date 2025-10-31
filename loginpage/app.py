from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, session, url_for
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
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
from datetime import timedelta
# Removed unused imports: FlaskForm, wtforms components, standalone bcrypt, flask_mysqldb

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
# This is the correct Bcrypt instance used in the signup/login routes
bcrypt = Bcrypt(app) 

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
      
    # Make sure all fields not sent during signup are explicitly nullable
    name = db.Column(db.String(100), nullable=True)
    
    # Pima features (set to nullable=True, removing conflicting DEFAULTs)
    pregnancies = db.Column(db.Integer, nullable=True)
    glucose = db.Column(db.Integer, nullable=True)
    blood_pressure = db.Column(db.Integer, nullable=True)
    skin_thickness = db.Column(db.Integer, nullable=True)
    insulin = db.Column(db.Integer, nullable=True)
    bmi = db.Column(db.Float, nullable=True)
    diabetes_pedigree_function = db.Column(db.Float, nullable=True)
    age = db.Column(db.Integer, nullable=True)
    last_diabetes_outcome = db.Column(db.Integer, nullable=True)
    
    # If you still want gender, make it nullable
    gender = db.Column(db.String(10), nullable=True)
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class HealthPlan(db.Model):
    __tablename__ = 'HealthPlan'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    diet_plan = db.Column(db.Text)
    exercise_plan = db.Column(db.Text)
    yoga_plan = db.Column(db.Text) # Added yoga_plan
    

with app.app_context():
    db.create_all()
    print("✅ Database tables created successfully")


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
    exercise_enc = joblib.load(exercise_enc_path)
    yoga_enc = joblib.load(yoga_enc_path)
else:
    healthplan_model = diet_enc = exercise_enc = yoga_enc = None

# ------------------- Page Routes -------------------

@app.route('/')
def home():
    """Default route → Login page (or dashboard if already logged in)"""
    if 'user_id' in session:
        if session.get('role') == 'patient':
            return redirect(url_for('uploadpage')) # Changed from 'upload' to 'uploadpage' for clarity
        elif session.get('role') == 'doctor':
            return redirect(url_for('doctor_dashboard'))
    return render_template('loginpage.html') # Assume this is the starting page

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serves static image files."""
    return send_from_directory('images', filename)

@app.route('/upload')
@login_required(role='patient')
def uploadpage():
    """Route for doctors to upload datasets for model training."""
    return render_template('upload.html')

@app.route('/premium')
@login_required(role='patient')
def premium():
    """Prediction input page for patients."""
    return render_template('premium.html')

@app.route('/dashboard')
@login_required(role='patient')
def dashboard():
    """Patient dashboard."""
    return render_template('dashboard.html')

@app.route('/doctor_dashboard')
@login_required(role='doctor')
def doctor_dashboard():
    """Doctor dashboard (separate route for doctor view)."""
    # Note: You would likely list patients here for the doctor
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
    return redirect(url_for('home')) # Redirect to home/login page

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
        # MODIFICATION: Redirect patient to uploadpage as requested
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
        return jsonify({'success': 'Login successful', 'redirect': redirect_url}), 200
    else:
        # Note: This returns JSON (401), not HTML.
        return jsonify({'error': 'Invalid email or password'}), 401
    
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
def predict():
    try:
        data = request.get_json()
        features = np.array([
            data['Pregnancies'], data['Glucose'], data['BloodPressure'],
            data['SkinThickness'], data['Insulin'], data['BMI'],
            data['DiabetesPedigreeFunction'], data['Age']
        ]).reshape(1, -1)
        
        # Save features to session before prediction for HealthPlan generation later
        session['last_prediction_data'] = {
            'Pregnancies': data['Pregnancies'], 'Glucose': data['Glucose'], 'BloodPressure': data['BloodPressure'],
            'SkinThickness': data['SkinThickness'], 'Insulin': data['Insulin'], 'BMI': data['BMI'],
            'DiabetesPedigreeFunction': data['DiabetesPedigreeFunction'], 'Age': data['Age'], 
            'Outcome': None # Outcome is determined by the prediction itself
        }

        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Update session with the prediction outcome
        session['last_prediction_data']['Outcome'] = int(prediction)

        advice = "High risk of diabetes detected. Consult your doctor soon." if prediction==1 else "Low risk of diabetes. Maintain healthy lifestyle."

        return jsonify({"prediction": int(prediction), "advice": advice})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['GET'])
def block_predict():
    return redirect(url_for('premium'))

# ------------------- HEALTH PLAN API HELPERS -------------------

def get_latest_patient_features(patient_id):
    """
    Fetches the last set of features submitted by the patient (uses session if available, 
    otherwise returns a default representative sample based on patient_id parity).
    """
    last_prediction_data = session.get('last_prediction_data')
    if last_prediction_data and last_prediction_data.get('Outcome') is not None:
          return last_prediction_data
          
    # Fallback to representative samples if no recent prediction data in session
    # Used for demonstration or when accessing healthplan without prior prediction
    print(f"⚠️ Using fallback features for patient {patient_id}")
    if patient_id % 2 == 1: # High-risk patient sample
        return {
            'Pregnancies': 6, 'Glucose': 148, 'BloodPressure': 72, 
            'SkinThickness': 35, 'Insulin': 0, 'BMI': 33.6, 
            'DiabetesPedigreeFunction': 0.627, 'Age': 50, 'Outcome': 1
        }
    # Low-risk patient sample
    return {
        'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66, 
        'SkinThickness': 29, 'Insulin': 0, 'BMI': 26.6, 
        'DiabetesPedigreeFunction': 0.351, 'Age': 31, 'Outcome': 0
    }

def get_saved_plan(patient_id):
    """Retrieves the last saved plan from the HealthPlan DB table."""
    plan_record = HealthPlan.query.filter_by(patient_id=patient_id).first()
    if plan_record and plan_record.diet_plan:
        return {
            # Plans are stored as multiline strings, split back into lists
            "diet": plan_record.diet_plan.split('\n'),
            "exercise": plan_record.exercise_plan.split('\n'),
            "yoga": plan_record.yoga_plan,
        }
    return None

def generate_ml_plan(features_dict):
    """Uses the trained MultiOutputClassifier to predict and decode the plan."""
    global healthplan_model, scaler, diet_enc, exercise_enc, yoga_enc
    
    if healthplan_model is None or scaler is None or diet_enc is None or exercise_enc is None or yoga_enc is None:
        raise Exception("ML Health Plan models/encoders are not fully loaded. Please upload and train models first.")

    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # 1. Prepare Features (PIMA features + Outcome)
    pima_features = [features_dict[name] for name in feature_names]
    outcome_feature = features_dict['Outcome']
    
    pima_array = np.array(pima_features).reshape(1, -1)
    
    # Scale Pima features (first 8)
    pima_scaled = scaler.transform(pima_array) 
    
    # Combine scaled PIMA features with the Outcome (the 9th feature, unscaled)
    final_input = np.hstack([pima_scaled, np.array([outcome_feature]).reshape(-1, 1)])

    # 2. Predict Plan Codes
    plan_codes = healthplan_model.predict(final_input)[0]
    
    # 3. Decode Plan Codes
    diet_plan_type = diet_enc.inverse_transform([plan_codes[0]])[0]
    exercise_intensity = exercise_enc.inverse_transform([plan_codes[1]])[0]
    yoga_routine = yoga_enc.inverse_transform([plan_codes[2]])[0]

    # 4. Map decoded type/intensity to actionable plan points
    # These static mappings provide the actionable text based on the ML prediction's category
    diet_map = {
        "Low-Carb": ["Limit starchy foods (rice, pasta, bread).", "Focus on lean protein and non-starchy vegetables.", "Track daily carbohydrate intake."],
        "Mediterranean": ["Prioritize fish, poultry, and legumes.", "Use olive oil for cooking.", "Eat fresh fruits and vegetables daily."],
        "Keto-like": ["Maintain a high-fat, very low-carb diet.", "Consult a doctor for regular monitoring.", "Avoid all sugars and grains."],
        "Standard": ["Control portion sizes.", "Reduce intake of processed foods and sodas.", "Eat meals at consistent times."],
    }
    exercise_map = {
        "High": ["45 mins intense cardio (running/swimming) 5x/week.", "Include 2 days of strength training.", "Monitor heart rate carefully."],
        "Moderate": ["30 mins brisk walking/light jogging 4x/week.", "Simple bodyweight exercises 3x/week.", "Stay active throughout the day."],
        "Low": ["20 mins light walking daily.", "Daily gentle stretching or mobility work.", "Avoid prolonged sitting."],
    }
    
    # Use the predicted categories to build the final plan structure
    return {
        "diet": diet_map.get(diet_plan_type, ["No specific plan recommendation."]),
        "exercise": exercise_map.get(exercise_intensity, ["No specific plan recommendation."]),
        "yoga": f"Recommended Yoga: **{yoga_routine}** (Focus on relaxation and core strength)"
    }

# ------------------- API Route for HEALTH PLAN -------------------
@app.route("/api/get_plan/<int:patient_id>")
@login_required(role='patient')
def api_get_plan(patient_id):
    """
    Fetches the patient's saved plan or generates a new one using the ML model, 
    then saves the new plan.
    """
    # Security check: Ensure patient ID from session matches the requested ID
    if session.get('user_id') != patient_id:
          return jsonify({"error": "Unauthorized access to patient plan."}), 403
          
    current_patient_id = patient_id
    recommend_new = request.args.get('recommend_new', 'false').lower() == 'true'

    try:
        # 1. Try to fetch the saved plan first (unless explicitly requesting a new one)
        if not recommend_new:
            saved_plan = get_saved_plan(current_patient_id)
            if saved_plan:
                return jsonify(saved_plan)

        # 2. Generate new plan using ML
        features = get_latest_patient_features(current_patient_id)
        
        ml_plan = generate_ml_plan(features)

        # 3. Save the newly generated plan to the database
        diet_str = "\n".join(ml_plan['diet'])
        exercise_str = "\n".join(ml_plan['exercise'])
        yoga_str = ml_plan['yoga']
        
        plan_record = HealthPlan.query.filter_by(patient_id=current_patient_id).first()
        if plan_record:
            # Update existing record
            plan_record.diet_plan = diet_str
            plan_record.exercise_plan = exercise_str
            plan_record.yoga_plan = yoga_str
        else:
            # Create new record
            plan_record = HealthPlan(patient_id=current_patient_id, diet_plan=diet_str, exercise_plan=exercise_str, yoga_plan=yoga_str)
            db.session.add(plan_record)
            
        db.session.commit()
        
        return jsonify(ml_plan)

    except Exception as e:
        db.session.rollback()
        print(f"Error in api_get_plan for patient {current_patient_id}: {e}")
        return jsonify({
            "error": "Could not generate or fetch plan.",
            "details": str(e)
        }), 500

# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(debug=True)

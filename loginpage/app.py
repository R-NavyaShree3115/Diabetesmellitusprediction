from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, session, url_for
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from flask_cors import CORS  # type: ignore
from flask_sqlalchemy import SQLAlchemy # type: ignore

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['HOME'] = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Globals to hold loaded model and scaler for prediction requests
model = None
scaler = None


@app.route('/')
def home():
    """Default route → Login page"""
    return render_template('/loginpage.html')

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('images', filename)

@app.route('/upload')
def uploadpage():
    return render_template('upload.html')

@app.route('/premium')
def premium():
    """After training → Prediction page"""
    return render_template('/premium.html')

@app.route('/dashboard')
def dashboard():
    """Optional dashboard route"""
    return render_template('/dashboard.html')

@app.route('/logout')
def logout():
    """Logout user"""
    return render_template('/logout.html')

@app.route('/healthplan')
def healthplan():
    """Health Plan page"""
    return render_template('/healthplan.html')
@app.route('/progress')
def progress():
    """Progress page"""
    return render_template('/progress.html')
@app.route('/alerts')
def alerts():
    """Alerts page"""
    return render_template('/alerts.html')
@app.route('/doctorconnect')
def doctorconnect():
    """Doctor Connect page"""
    return render_template('/doctorconnect.html')

# Example: Login button redirect
@app.route('/login', methods=['POST'])
def login():
    # (You can add login validation later)
    return redirect(url_for('uploadpage'))  # ✅ correct redirect

@app.route('/api/endpoint')
def api_endpoint():
    return jsonify({'message': 'API endpoint is working!'})



# -------------------------
# Route: Upload CSV and Train
# -------------------------
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'dataset' not in request.files:
        return jsonify({'status':'error','message':'No file uploaded!'})

    file = request.files['dataset']
    if file.filename == '':
        return jsonify({'status':'error','message':'No file selected!'})

    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    # Load CSV
    df = pd.read_csv(path)

    # Check target column
    if 'Outcome' not in df.columns:
        return jsonify({'status':'error','message':'CSV must contain "Outcome" column!'})

    if df['Outcome'].nunique() < 2:
        return jsonify({'status':'error','message':'Target column must have at least 2 classes!'})

    # Preview first 10 rows
    preview_html = df.head(10).to_html(classes='table table-striped', index=False)

    # -------------------------
    # Train Models
    # -------------------------
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
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

    # Voting Ensemble
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

    # Save models
    joblib.dump(logistic_model, os.path.join(MODEL_FOLDER,'logistic_model.pkl'))
    joblib.dump(decision_tree_model, os.path.join(MODEL_FOLDER,'decision_tree_model.pkl'))
    joblib.dump(random_forest_model, os.path.join(MODEL_FOLDER,'random_forest_model.pkl'))
    joblib.dump(svm_model, os.path.join(MODEL_FOLDER,'svm_model.pkl'))
    joblib.dump(knn_model, os.path.join(MODEL_FOLDER,'knn_model.pkl'))
    joblib.dump(ensemble_model, os.path.join(MODEL_FOLDER,'ensemble_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_FOLDER,'scaler.pkl'))

    # Evaluate
    accuracies = {
        'Logistic Regression': round(logistic_model.score(X_test_scaled, y_test)*100,2),
        'Decision Tree': round(decision_tree_model.score(X_test, y_test)*100,2),
        'Random Forest': round(random_forest_model.score(X_test, y_test)*100,2),
        'SVM': round(svm_model.score(X_test_scaled, y_test)*100,2),
        'KNN': round(knn_model.score(X_test_scaled, y_test)*100,2),
        'Ensemble': round(ensemble_model.score(X_test_scaled, y_test)*100,2)
    }

    best_model = max(accuracies, key=accuracies.get)
    joblib.dump(ensemble_model, os.path.join(MODEL_FOLDER, 'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_FOLDER, 'scaler.pkl'))

    return jsonify({
        'status':'success',
        'preview': preview_html,
        'accuracies': accuracies,
        'best_model': best_model,
        'message':'Models trained successfully!'
    })

model = joblib.load(os.path.join(MODEL_FOLDER, 'diabetes_model.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            data['Pregnancies'], data['Glucose'], data['BloodPressure'],
            data['SkinThickness'], data['Insulin'], data['BMI'],
            data['DiabetesPedigreeFunction'], data['Age']
        ]
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]

        if prediction == 1:
            advice = "High risk of diabetes detected. Consult your doctor soon."
        else:
            advice = "Low risk of diabetes. Keep maintaining a healthy lifestyle."

        return jsonify({
            "prediction": int(prediction),
            "advice": advice
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict', methods=['GET'])
def block_predict():
    # Prevents direct GET access to /predict
    return redirect(url_for('premium'))

db = SQLAlchemy(app)

# ================== DATABASE MODEL ==================
class HealthPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, nullable=False)
    diet_plan = db.Column(db.Text, nullable=True)
    exercise_plan = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())
db.create_all()

# ================== PATIENT HEALTH PLAN API (CSV BASED) ==================
@app.route("/api/healthplan")
def get_healthplan():
    try:
        patient_id = session.get('patient_id', 4000)
        diabetes_df = pd.read_csv("diabetes.csv")

        patient = diabetes_df[diabetes_df["id"] == patient_id]
        if patient.empty:
            return jsonify({"diet": ["No plan found"], "exercise": ["No plan found"], "yoga": "N/A"})

        row = patient.iloc[0]
        diet_items = [i.strip() for i in str(row["diet_plan"]).split("|")]
        exercise_items = [i.strip() for i in str(row["exercise_plan"]).split(";")]
        yoga = row["yoga_recommendation"]

        return jsonify({"diet": diet_items, "exercise": exercise_items, "yoga": yoga})
    except Exception as e:
        return jsonify({"error": str(e)})


# ================== DOCTOR UPDATE HEALTH PLAN (DB) ==================
@app.route("/api/update_plan", methods=["POST"])
def update_plan():
    data = request.json
    patient_id = data.get("patient_id")
    diet = "\n".join(data.get("diet", []))
    exercise = "\n".join(data.get("exercise", []))

    plan = HealthPlan.query.filter_by(patient_id=patient_id).first()
    if plan:
        plan.diet_plan = diet
        plan.exercise_plan = exercise
    else:
        plan = HealthPlan(patient_id=patient_id, diet_plan=diet, exercise_plan=exercise)
        db.session.add(plan)
    db.session.commit()
    return jsonify({"message": "Health plan updated successfully"})

# -------------------------
if __name__ == '__main__':
    app.run(debug=True)

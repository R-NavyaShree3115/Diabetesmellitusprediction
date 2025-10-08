from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# ======================
# Render Upload/Login Page
# ======================
@app.route("/")
def index():
    return render_template("upload.html")  # Make sure this file is in templates/


# ======================
# Handle Upload + Train
# ======================
@app.route("/login", methods=["POST"])
def login():
    try:
        username = request.form.get("username")
        password = request.form.get("password")
        file = request.files.get("dataset")

        if username == "admin" and password == "admin":
            if file and file.filename.endswith(".csv"):
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                print("✅ CSV saved at:", filepath)

                accuracy = train_model(filepath)
                flash(f"✅ Model trained successfully with accuracy: {accuracy:.2f}%", "success")
                return redirect(url_for("success"))
            else:
                flash("❌ Please upload a valid CSV file.", "error")
                return redirect(url_for("index"))
        else:
            flash("❌ Invalid username or password!", "error")
            return redirect(url_for("index"))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Internal Server Error: {str(e)}", 500


# ======================
# Train ML Model Function
# ======================
def train_model(csv_path):
    df = pd.read_csv(csv_path)
    required_columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    X = df[required_columns[:-1]]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    model_path = os.path.join(MODEL_FOLDER, "diabetes_model.pkl")
    joblib.dump(model, model_path)

    print(f"✅ Model trained and saved: {model_path} (Accuracy: {accuracy:.2f}%)")
    return accuracy


# ======================
# Success Page
# ======================
@app.route("/success")
def success():
    return "<h2>🎉 Login successful, dataset uploaded, and model trained!</h2>"


if __name__ == "__main__":
    app.run(debug=True)

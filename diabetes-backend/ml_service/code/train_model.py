# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

# ------------------------------
# 1. Load Dataset (default)
# ------------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'
]
data = pd.read_csv(url, names=columns)

# ------------------------------
# 2. Train & Save Function
# ------------------------------
def train_and_save_models(df=data):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling for LR, SVM, KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Individual models
    logistic_model = LogisticRegression().fit(X_train_scaled, y_train)
    decision_tree_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    svm_model = SVC(probability=True, random_state=42).fit(X_train_scaled, y_train)
    knn_model = KNeighborsClassifier(n_neighbors=5).fit(X_train_scaled, y_train)

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
    ).fit(X_train_scaled, y_train)

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(logistic_model, "models/logistic_model.pkl")
    joblib.dump(decision_tree_model, "models/decision_tree_model.pkl")
    joblib.dump(random_forest_model, "models/random_forest_model.pkl")
    joblib.dump(svm_model, "models/svm_model.pkl")
    joblib.dump(knn_model, "models/knn_model.pkl")
    joblib.dump(ensemble_model, "models/ensemble_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Models trained and saved successfully!")

# ------------------------------
# Train models on run
# ------------------------------
if __name__ == "__main__":
    train_and_save_models()

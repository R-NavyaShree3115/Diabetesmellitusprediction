from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_FOLDER = 'models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# -------------------------
# Route: Home Page
# -------------------------
@app.route('/')
def home():
    return render_template('upload.html')

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

    return jsonify({
        'status':'success',
        'preview': preview_html,
        'accuracies': accuracies,
        'best_model': best_model,
        'message':'Models trained successfully!'
    })

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# 1️⃣ Upload CSV & preview
# -----------------------------
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        if 'dataset' not in request.files:
            return jsonify({'status':'error','message':'No file uploaded'})
        file = request.files['dataset']
        if file.filename == '':
            return jsonify({'status':'error','message':'No file selected'})

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        try:
            df = pd.read_csv(path)
        except Exception as e:
            return jsonify({'status':'error','message': f'Failed to read CSV: {str(e)}'})

        preview_html = df.head(10).to_html(classes='table table-striped', index=False)
        return jsonify({'status':'success','preview': preview_html,'file_path': path})

    except Exception as e:
        return jsonify({'status':'error','message': f'Server error: {str(e)}'})

# -----------------------------
# 2️⃣ Train Models
# -----------------------------
@app.route('/train_models', methods=['POST'])
def train_models():
    try:
        file_path = request.json.get('file_path')
        if not file_path:
            return jsonify({'status':'error','message':'No file path provided'})

        df = pd.read_csv(file_path)

        # Example: assume target column is 'Outcome'
        if 'Outcome' not in df.columns:
            return jsonify({'status':'error','message':'CSV must have "Outcome" column as target'})

        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=500),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier()
        }

        results = {}
        best_model = None
        best_score = 0
        best_model_name = ""

        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test) * 100
            results[name] = round(score, 2)
            if score > best_score:
                best_score = score
                best_model = model
                best_model_name = name

        # Save the best model
        joblib.dump(best_model, 'diabetes_model.pkl')

        return jsonify({
            'status':'success',
            'results': results,
            'best_model': best_model_name,
            'message':'✅ Models trained successfully!'
        })

    except Exception as e:
        return jsonify({'status':'error','message': f'Training failed: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)

# models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_models_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    # Basic preprocessing
    df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']] = \
        df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.nan)
    
    # Fill missing values with median per Outcome
    columns = df.columns.drop('Outcome')
    for col in columns:
        medians = df.groupby('Outcome')[col].median()
        df.loc[(df['Outcome']==0) & (df[col].isnull()), col] = medians[0]
        df.loc[(df['Outcome']==1) & (df[col].isnull()), col] = medians[1]
    
    # Features & target
    y = df['Outcome']
    X = df.drop(['Outcome'], axis=1)
    
    # One-hot encoding for BMI/Insulin/Glucose categories (optional, or just numeric features)
    # Skipping for simplicity in Flask integration
    
    # Scaling
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # StandardScaler for some models
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, C=10, gamma=0.01),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(criterion='entropy', max_depth=15, max_features=0.75,
                                                min_samples_leaf=2, min_samples_split=3, n_estimators=130),
        'Gradient Boosting': GradientBoostingClassifier(learning_rate=0.1, loss='exponential', n_estimators=150),
        'XGBoost': XGBClassifier(objective='binary:logistic', learning_rate=0.01, max_depth=10, n_estimators=180) # pyright: ignore[reportUndefinedVariable]
    }
    
    results = {}
    best_model = None
    best_acc = 0
    best_name = ''
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)*100
        results[name] = round(acc,2)
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
    
    # Save the best model
    joblib.dump(best_model, 'diabetes_model.pkl')
    
    return results, best_name

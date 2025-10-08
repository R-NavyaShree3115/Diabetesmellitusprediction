import pandas as pd
import numpy as np

# ------------------------------
# CONFIGURATION
# ------------------------------
NUM_RECORDS = 100   # 🔢 Change this to 500, 1000 etc. if you want a larger dataset
OUTPUT_FILE = "diabetes_dataset.csv"

# ------------------------------
# GENERATE RANDOM DATA
# ------------------------------
np.random.seed(42)  # ✅ For reproducibility

# Create a dictionary with realistic ranges for each feature
data = {
    "Pregnancies": np.random.randint(0, 15, NUM_RECORDS),
    "Glucose": np.random.randint(70, 200, NUM_RECORDS),             # mg/dL
    "BloodPressure": np.random.randint(50, 122, NUM_RECORDS),       # mm Hg
    "SkinThickness": np.random.randint(10, 60, NUM_RECORDS),        # mm
    "Insulin": np.random.randint(15, 276, NUM_RECORDS),             # μU/mL
    "BMI": np.round(np.random.uniform(18.0, 45.0, NUM_RECORDS), 1), # kg/m²
    "DiabetesPedigreeFunction": np.round(np.random.uniform(0.1, 2.5, NUM_RECORDS), 2),
    "Age": np.random.randint(21, 80, NUM_RECORDS),
    "Outcome": np.random.randint(0, 2, NUM_RECORDS)                 # 0 = Non-diabetic, 1 = Diabetic
}

# ------------------------------
# CREATE DATAFRAME
# ------------------------------
df = pd.DataFrame(data)

# ------------------------------
# SAVE TO CSV
# ------------------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Dataset generated successfully: {OUTPUT_FILE}")
print(df.head(10))  # Optional: show first 10 rows

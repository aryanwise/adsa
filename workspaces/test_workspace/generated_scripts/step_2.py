import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('data/active/copy_test_data.csv')

# Separate target column
y_raw = df['CUSTOMER_SEGMENT']
X_raw = df.drop('CUSTOMER_SEGMENT', axis=1)

# Fit LabelEncoder on target
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Serialize LabelEncoder
joblib.dump(le, 'artifacts/label_encoder.joblib')

print(f"STATE shape={X_raw.shape} columns={list(X_raw.columns)}")
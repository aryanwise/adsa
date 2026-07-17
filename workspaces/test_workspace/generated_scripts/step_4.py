import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import joblib

# Load engineered features
engineered_features = pd.read_parquet('artifacts/engineered_features.parquet')

# Load label encoder and encoded labels
from sklearn.preprocessing import LabelEncoder
label_encoder = joblib.load('artifacts/label_encoder.joblib')
y = label_encoder.transform(engineered_features['CUSTOMER_SEGMENT'])

# Drop CUSTOMER_SEGMENT from engineered features
X = engineered_features.drop('CUSTOMER_SEGMENT', axis=1)

# Split data into training pool and hold-out set
sss = StratifiedShuffleSplit(test_size=0.20, random_state=42)
for train_idx, _ in sss.split(X, y):
    X_train, y_train = X.iloc[train_idx], y[train_idx]

# Define preprocessing pipeline
numeric_features = ['TOTAL_REVENUE_USD', 'ITEM_COUNT', 'MONTH_NUM']
numeric_transformer = Pipeline(steps=[
    ('log1p', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)])

# Fit the pipeline on the training pool
preprocessor.fit(X_train)

# Serialize the preprocessor
joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')
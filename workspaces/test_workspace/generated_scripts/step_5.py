import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import load

# Load engineered features and label encoder
engineered_features = pd.read_parquet('artifacts/engineered_features.parquet')
label_encoder = load('artifacts/label_encoder.joblib')

# Encode the target column
y_raw = engineered_features['CUSTOMER_SEGMENT']
y = label_encoder.transform(y_raw)

# Perform stratified hold-out split
sss = StratifiedShuffleSplit(test_size=0.2, random_state=42)
for train_index, holdout_index in sss.split(engineered_features.drop('CUSTOMER_SEGMENT', axis=1), y):
    break

# Save indices to files
train_idx = pd.Series(train_index, name='train_idx')
holdout_idx = pd.Series(holdout_index, name='holdout_idx')

train_idx.to_csv('artifacts/train_idx.npy', header=False, index=False)
holdout_idx.to_csv('artifacts/holdout_idx.npy', header=False, index=False)

print(f"STATE shape={engineered_features.shape} columns={list(engineered_features.columns)}")
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import f1_score
import joblib

# 1. Load the ENGINEERED dataset and preprocessors
df = pd.read_parquet('artifacts/engineered_features.parquet')
preprocessor = joblib.load('artifacts/preprocessor.joblib')
label_encoder = joblib.load('artifacts/label_encoder.joblib')

# 2. Separate Target and Features
X = df.drop(columns=['CUSTOMER_SEGMENT'])
y_raw = df['CUSTOMER_SEGMENT']
y_encoded = label_encoder.transform(y_raw)

# 3. Safely recreate the 80% train split natively to bypass the broken .npy file
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 4. Define Models (Including XGBoost)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
}

# 5. Stratified 5-Fold CV
cv_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Reset index for safe iloc splitting
X_train_base = X_train.reset_index(drop=True)

for name, model in models.items():
    fold_scores = []
    
    for train_idx, val_idx in skf.split(X_train_base, y_train):
        # Split folds
        X_f_train, X_f_val = X_train_base.iloc[train_idx], X_train_base.iloc[val_idx]
        y_f_train, y_f_val = y_train[train_idx], y_train[val_idx]
        
        # Transform data
        X_f_train_tf = preprocessor.transform(X_f_train)
        X_f_val_tf = preprocessor.transform(X_f_val)
        
        # Train & Predict
        model.fit(X_f_train_tf, y_f_train)
        preds = model.predict(X_f_val_tf)
        fold_scores.append(f1_score(y_f_val, preds, average='macro'))
        
    cv_results.append((name, np.mean(fold_scores)))

# 6. Save results
cv_df = pd.DataFrame(cv_results, columns=['Model', 'Mean Macro F1'])
cv_df.to_csv('artifacts/model_cv_report.csv', index=False)

print(f"✅ CV Completed. Results:\n{cv_df}")
print(f"STATE shape={df.shape} columns={list(df.columns)}")
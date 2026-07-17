import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import f1_score, confusion_matrix
import joblib

# 1. Load engineered data and preprocessors
df = pd.read_parquet('artifacts/engineered_features.parquet')
preprocessor = joblib.load('artifacts/preprocessor.joblib')
label_encoder = joblib.load('artifacts/label_encoder.joblib')

# 2. Recreate splits safely (80/20 Holdout)
X = df.drop(columns=['CUSTOMER_SEGMENT'])
y = label_encoder.transform(df['CUSTOMER_SEGMENT'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Read the CV report from Step 6 to find the best model
cv_report = pd.read_csv('artifacts/model_cv_report.csv')
best_model_name = cv_report.loc[cv_report['Mean Macro F1'].idxmax(), 'Model']
print(f"🏆 Best Model Selected: {best_model_name}")

# 4. Instantiate the winning model
if best_model_name == 'Logistic Regression':
    model = LogisticRegression(max_iter=1000)
elif best_model_name == 'Random Forest':
    model = RandomForestClassifier(random_state=42)
else:
    model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

# 5. Transform data and train the final model
X_train_tf = preprocessor.transform(X_train)
X_test_tf = preprocessor.transform(X_test)
model.fit(X_train_tf, y_train)

# 6. Evaluate on the Holdout Set
y_pred = model.predict(X_test_tf)
final_f1 = f1_score(y_test, y_pred, average='macro')
print(f"📊 Final Holdout Macro F1 Score: {final_f1:.4f}")

# 7. Generate and save Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('artifacts/confusion_matrix.png')
print("✅ Confusion matrix saved to artifacts/confusion_matrix.png")

# 8. Save the winning model for Step 8 to package
joblib.dump(model, 'artifacts/best_model.joblib')
print("✅ Final model saved to artifacts/best_model.joblib")

print(f"STATE shape={df.shape} columns={list(df.columns)}")
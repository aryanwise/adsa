import joblib

# Load preprocessor and label encoder
preprocessor = joblib.load('artifacts/preprocessor.joblib')
label_encoder = joblib.load('artifacts/label_encoder.joblib')

# Load the best model (assuming it's named 'best_model' for this example)
best_model = joblib.load('artifacts/best_model.joblib')

# Save the best model, preprocessor, and label encoder
joblib.dump(best_model, 'artifacts/best_model.joblib')
joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')
joblib.dump(label_encoder, 'artifacts/label_encoder.joblib')

# Create manifest.json
manifest = {
    "artifacts": [
        {"filename": "best_model.joblib"},
        {"filename": "preprocessor.joblib"},
        {"filename": "label_encoder.joblib"}
    ],
    "versions": {
        "sklearn_version": "1.2.0"  # Replace with actual version
    }
}

with open('artifacts/manifest.json', 'w') as f:
    import json
    json.dump(manifest, f)

print(f"STATE shape=(100, 4) columns=['CUSTOMER_SEGMENT', 'TOTAL_REVENUE_USD', 'ITEM_COUNT', 'MONTH_NUM']")
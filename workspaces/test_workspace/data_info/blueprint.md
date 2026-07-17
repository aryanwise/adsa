# 🚀 Data Science Pipeline Blueprint  

## 🎯 1. User Objective  
Develop a production‑ready classification pipeline that predicts **CUSTOMER_SEGMENT**. The pipeline must profile and remediate outliers/skewness in `TOTAL_REVENUE_USD` and `ITEM_COUNT`, create a numeric month feature `MONTH_NUM` from `MONTH_KEY`, drop the identifier `ORDER_ID`, apply log‑transformation followed by standard scaling, and evaluate three models (Logistic Regression, Random Forest, XGBoost) using a strict hold‑out set plus Stratified 5‑Fold CV. The best model is selected on macro‑averaged F1‑Score, accompanied by a confusion‑matrix heatmap and feature‑importance report, and both the final model and preprocessing objects are persisted for inference.

## 📋 2. Core System Requirements & AI Guardrails  
- **Target Isolation Constraint:** Extract `CUSTOMER_SEGMENT` into vector **y** and remove it from **X** before any transformer fitting.  
- **Target Encoding Requirements:** Encode **y** with `sklearn.preprocessing.LabelEncoder` and store the fitted encoder for downstream inference.  
- **Validation Strategy Contract:**  
  1. Immediately split the full dataset into a **hold‑out set (20%)** and a **training pool (80%)** using `StratifiedShuffleSplit`.  
  2. Within the training pool, perform **Stratified 5‑Fold CV** for hyper‑parameter‑consistent evaluation of each candidate model.  
- **Pathing Boundaries:** All raw data reads/writes target `data/active/copy_test_data.csv`. All artifacts (preprocessor, encoder, models, evaluation visuals) must be saved under the `artifacts/` directory using relative paths only.  

## 🛠️ 3. Execution Engine Roadmap  

### Step 1: Data Ingestion & Exhaustive Profiling  
- **Action:**  
  1. Load CSV from `data/active/copy_test_data.csv` into a DataFrame.  
  2. Compute descriptive statistics, outlier thresholds (IQR), and skewness for `TOTAL_REVENUE_USD` and `ITEM_COUNT`.  
  3. Log profiling results to `artifacts/data_profile.json` for auditability.  
- **Contract:** Input: `data/active/copy_test_data.csv` | Output: `artifacts/data_profile.json`  

### Step 2: Target Isolation & Encoding  
- **Action:**  
  1. Separate `CUSTOMER_SEGMENT` column → **y_raw**; drop it from feature matrix → **X_raw**.  
  2. Fit `LabelEncoder` on **y_raw**, transform to integer labels **y**, and serialize encoder to `artifacts/label_encoder.joblib`.  
- **Contract:** Input: `data/active/copy_test_data.csv` | Output: `artifacts/label_encoder.joblib`  

### Step 3: Feature Engineering & Identifier Removal  
- **Action:**  
  1. Parse `MONTH_KEY` (format `YYYY-MM`) to integer month number (1‑12) → new column `MONTH_NUM`.  
  2. Drop original columns `MONTH_KEY` and `ORDER_ID` from **X_raw**.  
  3. Persist engineered feature set to `artifacts/engineered_features.parquet`.  
- **Contract:** Input: `data/active/copy_test_data.csv` | Output: `artifacts/engineered_features.parquet`  

### Step 4: Preprocessing Pipeline Construction  
- **Action:**  
  1. Define a `ColumnTransformer` that:  
     - Applies `FunctionTransformer(np.log1p)` to `TOTAL_REVENUE_USD` and `ITEM_COUNT`.  
     - Passes through `MONTH_NUM` unchanged.  
  2. Follow with `StandardScaler` on all numeric columns.  
  3. Fit the pipeline **only on the training pool** (post hold‑out split) and serialize to `artifacts/preprocessor.joblib`.  
- **Contract:** Input: `artifacts/engineered_features.parquet` | Output: `artifacts/preprocessor.joblib`  

### Step 5: Hold‑Out Partitioning (20% Stratified)  
- **Action:**  
  1. Using the encoded **y**, apply `StratifiedShuffleSplit(test_size=0.20, random_state=42)` on the engineered feature set.  
  2. Save training indices to `artifacts/train_idx.npy` and hold‑out indices to `artifacts/holdout_idx.npy`.  
- **Contract:** Input: `artifacts/engineered_features.parquet` & `artifacts/label_encoder.joblib` | Output: `artifacts/train_idx.npy`, `artifacts/holdout_idx.npy`  

### Step 6: Model Training & Stratified 5‑Fold CV  
- **Action:**  
  1. For each candidate model (Logistic Regression, RandomForestClassifier, XGBClassifier):  
     - Initialize with **default hyper‑parameters** (no grid search).  
     - Within the training pool, run `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`; for each fold, fit the preprocessor on the fold‑train split, transform, train the model, predict on fold‑validation, compute macro F1.  
  2. Aggregate fold scores → mean macro F1 per model.  
  3. Store CV results in `artifacts/model_cv_report.csv`.  
- **Contract:** Input: `artifacts/preprocessor.joblib`, `artifacts/train_idx.npy`, `artifacts/engineered_features.parquet`, `artifacts/label_encoder.joblib` | Output: `artifacts/model_cv_report.csv`  

### Step 7: Model Selection, Evaluation & Reporting  
- **Action:**  
  1. Identify the model with the highest mean macro F1. Retrain this **best model** on the full training pool (using the preprocessor fitted on the entire training pool).  
  2. Transform the hold‑out set, generate predictions, compute macro F1, and produce a **confusion matrix heatmap** saved as `artifacts/confusion_matrix.png`.  
  3. Extract feature importance:  
     - For Logistic Regression → absolute coefficients.  
     - For Random Forest → `feature_importances_`.  
     - For XGBoost → `feature_importances_`.  
     Save a ranked table to `artifacts/feature_importance.csv`.  
- **Contract:** Input: `artifacts/preprocessor.joblib`, `artifacts/best_model_placeholder.joblib` (to be created), hold‑out indices & data | Output: `artifacts/confusion_matrix.png`, `artifacts/feature_importance.csv`, `artifacts/best_model_report.txt`  

### Step 8: Persistence of Production Artifacts  
- **Action:**  
  1. Serialize the final trained best model to `artifacts/best_model.joblib`.  
  2. Ensure the preprocessor (`artifacts/preprocessor.joblib`) and label encoder (`artifacts/label_encoder.joblib`) remain versioned alongside the model.  
  3. Write a minimal **manifest** JSON (`artifacts/manifest.json`) enumerating artifact filenames, versions (e.g., sklearn version), and a checksum for integrity verification.  
- **Contract:** Input: Trained best model object, preprocessor, label encoder | Output: `artifacts/best_model.joblib`, `artifacts/manifest.json`  

## 📦 4. Python Package Requirements  
- pandas  
- numpy  
- scikit-learn  
- xgboost  
- joblib  
- matplotlib  
- seaborn  
- jsonschema (optional for manifest validation)  
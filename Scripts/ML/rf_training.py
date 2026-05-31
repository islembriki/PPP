import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ============================================
# 1. LOAD
# ============================================
log("Loading fixed dataset...")
df = pd.read_csv(r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Applying math transformations...")
df['RMS']      = np.sqrt(np.abs(df['Variance']))
df['Abs_Mean'] = np.abs(df['Mean'])
df['Log_PAPR'] = np.log10(df['PAPR'] + 1e-9)

# ============================================
# 2. TARGET PREP (TYPE ONLY)
# ============================================
X = df.drop(['Label', 'Mode'], axis=1)
y = df['Label']

target_names = ['Background', 'Bebop', 'AR_Drone', 'Phantom']

# ============================================
# 3. SPLIT & SCALE
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
log("✓ Data Scaled")

# ============================================
# 4. TRAIN
# ============================================
log("⏳ Training Deep Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf.fit(X_train_scaled, y_train)

# ============================================
# 5. EVALUATE
# ============================================
y_pred = rf.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print("🏆 FINAL RESULTS: DRONE TYPE IDENTIFICATION")
print("="*60)
print(f"ACCURACY: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ============================================
# 6. SAVE PKL (unchanged from your original)
# ============================================
joblib.dump(rf,     'drone_type_model_final.pkl')
joblib.dump(scaler, 'scaler_final.pkl')
log("✓ Model and Scaler saved.")

# ============================================
# 7. SAVE RESULTS TO JSON  ← NEW, no impact on accuracy
# ============================================
log("Saving results to rf_results.json ...")

# classification_report with output_dict=True gives per-class numbers cleanly
report_dict = classification_report(
    y_test, y_pred,
    target_names=target_names,
    output_dict=True          # returns a dict instead of a string
)

# Confusion matrix as a plain nested list (JSON-serialisable)
cm = confusion_matrix(y_test, y_pred).tolist()

# Feature importances paired with their column names
feature_importance = dict(
    zip(X.columns.tolist(), rf.feature_importances_.tolist())
)
# Sort descending so the dashboard can display a ranked bar chart
feature_importance_sorted = dict(
    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
)

results = {
    "trained_at":          datetime.now().isoformat(),
    "model":               "RandomForestClassifier",
    "n_estimators":        300,
    "class_weight":        "balanced",
    "test_size":           0.2,
    "random_state":        42,
    "accuracy":            round(acc * 100, 4),      # e.g. 82.34
    "target_names":        target_names,
    "classification_report": report_dict,             # per-class precision/recall/f1
    "confusion_matrix":    cm,                        # rows = actual, cols = predicted
    "feature_importances": feature_importance_sorted, # column_name -> importance score
    "dataset_shape": {
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "n_features":    int(X.shape[1])
    }
}

with open('rf_results.json', 'w') as f:
    json.dump(results, f, indent=2)

log("✓ rf_results.json saved — ready to load into your dashboard!")
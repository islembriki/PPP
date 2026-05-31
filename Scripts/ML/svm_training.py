import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Changement : SVC au lieu de KNN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ============================================
# 1. PATHS
# ============================================
DATASET_PATH = r"C:\Users\garba\Desktop\PPP FINAL\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"
SAVE_DIR = r"C:\Users\garba\Desktop\PPP FINAL\PPP\ml_trained_models_type_only"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ============================================
# 2. LOAD & FEATURE ENGINEERING
# ============================================
log("🚀 Starting Optimized SVM Training...")
df = pd.read_csv(DATASET_PATH)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Applying Feature Engineering (The 82% Logic)...")
df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Log_Var', 'PAPR_Mean', 'Kurt_Skew']
X = df[feature_cols]
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
X_test_scaled = scaler.transform(X_test)
log("✓ Data Engineering and Scaling complete.")

# ============================================
# 4. SVM TRAINING 
# ============================================
log("⏳ Training SVM... (This may take a moment)")
# Utilisation d'un noyau RBF qui est le plus performant pour ce type de données
# C=1.0 et gamma='scale' sont les standards
model_svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
model_svm.fit(X_train_scaled, y_train)

# ============================================
# 5. RESULTS & SAVE
# ============================================
log("⏳ Predicting...")
y_pred = model_svm.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*40)
print(f"🔥 FINAL SVM ACCURACY: {acc*100:.2f}%")
print("="*40)
print(report)

# --- SAVE PKL (Noms de fichiers mis à jour pour SVM) ---
joblib.dump(model_svm, os.path.join(SAVE_DIR, "svm_model.pkl"))
joblib.dump(scaler,    os.path.join(SAVE_DIR, "svm_scaler.pkl"))

# --- SAVE JSON (Structure strictement identique) ---
results_json = {
    "trained_at":            datetime.now().isoformat(),
    "model_name":            "Support Vector Machine",
    "kernel":                "rbf",
    "C":                     1.0,
    "test_size":             0.2,
    "random_state":          42,
    "features_used":         feature_cols,
    "accuracy":              round(float(acc) * 100, 4),
    "target_names":          target_names,
    "classification_report": classification_report(
                                 y_test, y_pred,
                                 target_names=target_names,
                                 output_dict=True
                             ),
    "confusion_matrix":      cm.tolist(),
    "dataset_shape": {
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "n_features":    int(X.shape[1])
    }
}

with open(os.path.join(SAVE_DIR, "svm_results.json"), 'w') as f:
    json.dump(results_json, f, indent=2)

log("✅ All files saved to ml_trained_models_type_only")
import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ============================================
# 1. PATHS
# ============================================
DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"
SAVE_DIR = r"C:\Users\HP\Desktop\PPP\ml_trained_models_type_only"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ============================================
# 2. LOAD & FEATURE ENGINEERING (The Secret Sauce)
# ============================================
log("🚀 Starting Optimized KNN Training...")
df = pd.read_csv(DATASET_PATH)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Applying Feature Engineering (The 82% Logic)...")
# We add these because they separate the Parrot drones (Bebop/AR) much better
df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

# Now we have 8 features instead of 5
feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Log_Var', 'PAPR_Mean', 'Kurt_Skew']
X = df[feature_cols]
y = df['Label']

target_names = ['Background', 'Bebop', 'AR_Drone', 'Phantom']

# ============================================
# 3. SPLIT & SCALE (CRITICAL)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
log("✓ Data Engineering and Scaling complete.")

# ============================================
# 4. KNN TRAINING (Using k=3 for sharper boundaries)
# ============================================
log("⏳ Training KNN... (This will take a few minutes)")
# k=3 often works better for overlapping classes like Bebop/AR
knn = KNeighborsClassifier(n_neighbors=52, weights='distance', n_jobs=-1)
knn.fit(X_train_scaled, y_train)

# ============================================
# 5. RESULTS & SAVE
# ============================================
log("⏳ Predicting...")
y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*40)
print(f"🔥 FINAL KNN ACCURACY: {acc*100:.2f}%")
print("="*40)
print(report)

# --- SAVE PKL ---
joblib.dump(knn,    os.path.join(SAVE_DIR, "knn_model.pkl"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "knn_scaler.pkl"))

# --- SAVE JSON ---
# classification_report called twice on purpose:
#   first call (above) → string for the terminal print
#   second call (below) → dict for the JSON, zero impact on accuracy
results_json = {
    "trained_at":            datetime.now().isoformat(),
    "model_name":            "K-Nearest Neighbors",
    "n_neighbors":           52,
    "weights":               "distance",
    "test_size":             0.2,
    "random_state":          42,
    "features_used":         feature_cols,
    "accuracy":              round(float(acc) * 100, 4),
    "target_names":          target_names,
    "classification_report": classification_report(
                                 y_test, y_pred,
                                 target_names=target_names,
                                 output_dict=True          # ← structured dict, not string
                             ),
    "confusion_matrix":      cm.tolist(),
    "dataset_shape": {
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "n_features":    int(X.shape[1])
    }
}

with open(os.path.join(SAVE_DIR, "knn_results.json"), 'w') as f:
    json.dump(results_json, f, indent=2)

log("✅ All files saved to ml_trained_models_type_only")
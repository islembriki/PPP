import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================
# 0. CONFIGURATION
# ============================================
SAVE_DIR = r"C:\Users\HP\Desktop\PPP\ml_trained_models_mode_included"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"

# ============================================
# 1. LOAD & HYPER-ENGINEERING
# ============================================
log("🚀 Starting High-Accuracy Mode Run...")
df = pd.read_csv(DATASET_PATH)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Applying Behavioral Feature Engineering...")
# 1. The original "Secret Sauce"
df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

# 2. NEW: Behavioral Interactions (To separate flight modes)
# Ratio of Peak power to stability (Separates Hovering from Flying)
df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + 1e-9)
# Energy indicator
df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
# Shape indicator
df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])
# Non-linear Mean (helps with tiny sensor offsets)
df['Mean_Cube'] = df['Mean']**3

# ============================================
# 2. TARGET: 10 CLASSES
# ============================================
df['Target'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)
X = df.drop(['Label', 'Mode', 'Target'], axis=1)
y = df['Target']
target_names = sorted(df['Target'].unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 3. HIGH-INTENSITY TRAINING
# ============================================
log(f"⏳ Training Extreme Forest (500 Trees) on {X.shape[1]} features...")
# We use 500 trees and deeper growth to find the tiny Parrot differences
rf = RandomForestClassifier(
    n_estimators=500,        
    max_depth=40,            
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced', 
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf.fit(X_train, y_train)

# ============================================
# 4. RESULTS & JSON
# ============================================
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\n🎯 NEW ACCURACY: {acc*100:.2f}%")

# Save JSON
results_json = {
    "model_name": "Random Forest (Optimized)",
    "accuracy": float(acc),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "target_names": target_names,
    "trained_at": datetime.now().isoformat()
}

with open(os.path.join(SAVE_DIR, "rf_mode.json"), 'w') as f:
    json.dump(results_json, f, indent=4)

joblib.dump(rf, os.path.join(SAVE_DIR, "rf_mode.pkl"))
log("✅ Done! If this is still < 75%, it proves the CNN is required.")
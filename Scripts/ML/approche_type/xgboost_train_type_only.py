import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================
# 0. CONFIGURATION
# ============================================
SAVE_DIR = r"C:\Users\HP\Desktop\PPP\ml_trained_models_type_only"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"

# ============================================
# 1. LOAD & HYPER-ENGINEERING
# ============================================
log("🚀 Starting XGBoost TYPE ONLY Run (GPU Acceleration)...")
df = pd.read_csv(DATASET_PATH)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Applying Hardware Signature Feature Engineering...")
eps = 1e-9
# Caractéristiques qui séparent bien les chipsets Parrot (Bebop/AR) des autres
df['Log_Var']   = np.log10(np.abs(df['Variance']) + eps)
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + eps)
df['Signal_Power'] = (df['Mean']**2) + df['Variance']
df['Var_PAPR_Ratio'] = df['Variance'] / (df['PAPR'] + eps)

# ============================================
# 2. TARGET: DRONE TYPE ONLY (4 Classes)
# ============================================
X = df.drop(['Label', 'Mode'], axis=1)
y = df['Label'] # 0, 1, 2, 3
target_names = ['Background', 'Bebop', 'AR_Drone', 'Phantom']

# Encodage (XGBoost veut 0, 1, 2, 3)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

log(f"Training on {len(X_train)} samples with {X.shape[1]} features.")

# ============================================
# 3. XGBOOST TRAINING (TUNED FOR GPU)
# ============================================
log("⏳ Training XGBoost on NVIDIA GPU...")

model = XGBClassifier(
    n_estimators=500,        # On augmente le nombre d'arbres pour le 90%+
    max_depth=8,             # Profondeur augmentée
    learning_rate=0.05,      # Apprentissage plus lent et plus précis
    subsample=0.9,           
    colsample_bytree=0.9,
    tree_method='hist',      # Algorithme rapide
    device='cuda',           # 🚀 UTILISE TA CARTE NVIDIA
    random_state=42,
    objective='multi:softprob',
    eval_metric='mlogloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# ============================================
# 4. EVALUATION
# ============================================
log("📊 Evaluating...")
y_pred = model.predict(X_test)

# Inversion de l'encodage pour le rapport
y_test_orig = le.inverse_transform(y_test)
y_pred_orig = le.inverse_transform(y_pred)

acc = accuracy_score(y_test_orig, y_pred_orig)
report_dict = classification_report(y_test_orig, y_pred_orig, target_names=target_names, output_dict=True)
cm = confusion_matrix(y_test_orig, y_pred_orig)

print(f"\n🎯 FINAL TYPE-ONLY ACCURACY: {acc*100:.2f}%")
print("\n📋 Detailed Report:")
print(classification_report(y_test_orig, y_pred_orig, target_names=target_names))

# ============================================
# 5. SAVE FOR DASHBOARD
# ============================================
results_json = {
    "model_name": "XGBoost (Type Only - GPU)",
    "accuracy": float(acc),
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "target_names": target_names,
    "trained_at": datetime.now().isoformat()
}

json_path = os.path.join(SAVE_DIR, "xgboost_type_results.json")
pkl_path  = os.path.join(SAVE_DIR, "xgboost_type_model.pkl")

with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=4)

joblib.dump(model, pkl_path)
log(f"✅ Success! Results saved in {SAVE_DIR}")
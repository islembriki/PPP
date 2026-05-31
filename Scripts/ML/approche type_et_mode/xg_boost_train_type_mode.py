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
SAVE_DIR = r"C:\Users\HP\Desktop\PPP\ml_trained_models_mode_included"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"

# ============================================
# 1. LOAD & HYPER-ENGINEERING
# ============================================
log("🚀 Starting XGBoost Mode Run...")
df = pd.read_csv(DATASET_PATH)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Applying Behavioral Feature Engineering...")
# 1. Original Sauce
df['Log_Var']   = np.log10(np.abs(df['Variance']) + 1e-9)
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

# 2. Behavioral Interactions
df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + 1e-9)
df['Signal_Energy']  = (df['Mean']**2) + df['Variance']
df['Shape_Factor']   = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])
df['Mean_Cube']      = df['Mean']**3

# ============================================
# 2. TARGET: 10 CLASSES
# ============================================
df['Target'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)
X = df.drop(['Label', 'Mode', 'Target'], axis=1)
y = df['Target']
target_names = sorted(df['Target'].unique())

# XGBoost nécessite des labels numériques
le = LabelEncoder()
y_encoded = le.fit_transform(y)

log(f"Dataset shape: {X.shape} | Classes: {len(target_names)} → {target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

log(f"Train: {len(X_train)} lignes | Test: {len(X_test)} lignes")

# ============================================
# 3. XGBOOST TRAINING
# ============================================
log(f"⏳ Training XGBoost on {X.shape[1]} features, {len(X_train)} samples...")

model = XGBClassifier(
    n_estimators=300,        # Nombre d'arbres
    max_depth=6,             # Profondeur max par arbre
    learning_rate=0.1,       # Pas d'apprentissage
    subsample=0.8,           # 80% des données par arbre (évite overfitting)
    colsample_bytree=0.8,    # 80% des features par arbre
    use_label_encoder=False,
    eval_metric='mlogloss',  # Métrique multiclasse
    tree_method='hist',      # ← CLEF : algorithme rapide basé histogramme
    device='cuda',           # ← GPU si disponible, sinon remplace par 'cpu'
    n_jobs=-1,               # Tous les CPU
    random_state=42,
    verbosity=1
)

# Early stopping pour éviter overfitting + gagner du temps
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50               # Affiche le score tous les 50 arbres
)

# ============================================
# 4. EVALUATION
# ============================================
log("📊 Évaluation sur le test set complet...")
y_pred = model.predict(X_test)

# Reconvertir les labels numériques → originaux
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

acc    = accuracy_score(y_test_labels, y_pred_labels)
report = classification_report(y_test_labels, y_pred_labels, target_names=target_names, output_dict=True)
cm     = confusion_matrix(y_test_labels, y_pred_labels)

print(f"\n🎯 XGBOOST ACCURACY: {acc*100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=target_names))

# ============================================
# 5. SAVE MODEL & JSON
# ============================================
results_json = {
    "model_name"            : "XGBoost (Histogram - GPU)",
    "accuracy"              : float(acc),
    "train_samples"         : len(X_train),
    "test_samples"          : len(X_test),
    "n_features"            : X.shape[1],
    "n_estimators"          : 300,
    "classification_report" : report,
    "confusion_matrix"      : cm.tolist(),
    "target_names"          : target_names,
    "trained_at"            : datetime.now().isoformat()
}

json_path = os.path.join(SAVE_DIR, "xgboost_mode.json")
pkl_path  = os.path.join(SAVE_DIR, "xgboost_mode.pkl")

with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=4)

joblib.dump(model, pkl_path)
joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder.pkl"))  # Sauvegarde le encoder aussi

log(f"✅ Done! Model saved to {pkl_path}")
log(f"📊 Stats saved to {json_path}")
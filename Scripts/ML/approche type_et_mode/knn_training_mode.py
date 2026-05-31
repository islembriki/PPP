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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================
# 0. CONFIGURATION
# ============================================
SAVE_DIR = r"C:\Users\HP\Desktop\PPP\ml_trained_models_mode_included"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

DATASET_PATH = r"C:\Users\HP\Desktop\PPP\processed data\ML\FINAL_GLOBAL_DRONE_DATASET.csv"

# ============================================
# 1. LOAD & ENHANCED ENGINEERING
# ============================================
log("🚀 Démarrage du KNN (Type + Mode)...")
df = pd.read_csv(DATASET_PATH)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Application de l'ingénierie comportementale...")
# On utilise la même logique que le RF pour que les modèles comparent les mêmes données
df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)
df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + 1e-9)
df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])

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
# 3. SCALING (OBLIGATOIRE POUR KNN)
# ============================================
log("Normalisation des données...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 4. TRAINING (k=5 avec poids par distance)
# ============================================
log("⏳ Entraînement du KNN (k=5, weights='distance')...")
# 'distance' donne plus de poids aux voisins les plus proches, utile pour les modes
knn = KNeighborsClassifier(n_neighbors=22, weights='distance', n_jobs=-1)
knn.fit(X_train_scaled, y_train)

# ============================================
# 5. EVALUATION
# ============================================
log("Prédiction en cours...")
y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\n📊 RÉSULTAT KNN (TYPE+MODE):")
print(f"Accuracy: {acc*100:.2f}%")
print("-" * 30)

# ============================================
# 6. SAVE (knn_mode.pkl + knn_mode.json)
# ============================================
# 1. Sauvegarde PKL et Scaler
joblib.dump(knn, os.path.join(SAVE_DIR, "knn_mode.pkl"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "knn_scaler_mode.pkl"))

# 2. Préparation du JSON
results_json = {
    "model_name": "K-Nearest Neighbors (Type + Mode)",
    "accuracy": float(acc),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "target_names": target_names,
    "trained_at": datetime.now().isoformat()
}

json_path = os.path.join(SAVE_DIR, "knn_mode.json")
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=4)

log(f"✅ KNN Results saved to: {json_path}")
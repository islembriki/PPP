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

# Fonction utilitaire pour afficher des messages horodatés dans la console
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# 0. CONFIGURATION DES CHEMINS RELATIFS

# Dossier où seront sauvegardés le modèle entraîné et les résultats JSON
SAVE_DIR = "./PPP/ml_trained_models_type_only"
# Création automatique du dossier s'il n'existe pas encore
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# Chemin relatif vers le fichier CSV contenant les données globales
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"

# 1. CHARGEMENT ET INGÉNIERIE AVANCÉE (HYPER-ENGINEERING)

log(" Démarrage de l'entraînement XGBoost TYPE ONLY (Accélération GPU)...")
# Chargement du dataset avec Pandas
df = pd.read_csv(DATASET_PATH)
# Nettoyage : remplacement des valeurs infinies par NaN et suppression des lignes vides
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Application de l'ingénierie des signatures matérielles...")
# Constante pour éviter la division par zéro ou les erreurs de logarithme
eps = 1e-9
# Caractéristiques créées pour mieux séparer les signatures radio des chipsets
# Logarithme de la variance pour stabiliser l'échelle des données
df['Log_Var']   = np.log10(np.abs(df['Variance']) + eps)
# Interaction entre le pic de puissance et la moyenne
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
# Ratio de forme du signal (Kurtosis / Skewness)
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + eps)
# Calcul de la puissance totale estimée du signal
df['Signal_Power'] = (df['Mean']**2) + df['Variance']
# Ratio entre la variance (stabilité) et le pic de puissance (PAPR)
df['Var_PAPR_Ratio'] = df['Variance'] / (df['PAPR'] + eps)

# 2. CIBLE : TYPE DE DRONE UNIQUEMENT

# X contient les variables explicatives (on retire les colonnes cibles)
X = df.drop(['Label', 'Mode'], axis=1)
# y contient la variable cible (le type de drone : 0, 1, 2 ou 3)
y = df['Label'] 
# Liste des noms des catégories pour l'affichage final
target_names = ['Background', 'Bebop', 'AR_Drone', 'Phantom']

# XGBoost nécessite des étiquettes numériques consécutives (0, 1, 2, 3)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Division des données en Train (80%) et Test (20%) avec maintien de l'équilibre des classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

log(f"Entraînement sur {len(X_train)} échantillons avec {X.shape[1]} caractéristiques.")

# 3. ENTRAÎNEMENT XGBOOST 
log(" Entraînement de XGBoost sur le GPU NVIDIA...")

# Configuration du classifieur XGBoost :
# n_estimators=500 : nombre élevé d'arbres pour maximiser la précision
# max_depth=8 : profondeur des arbres augmentée pour capturer des motifs complexes
# learning_rate=0.05 : apprentissage lent pour une meilleure convergence
# tree_method='hist' : algorithme rapide basé sur des histogrammes
# device='cuda' :  INDIQUE À XGBOOST D'UTILISER LA CARTE GRAPHIQUE NVIDIA
model = XGBClassifier(
    n_estimators=500,        
    max_depth=8,             
    learning_rate=0.05,      
    subsample=0.9,           
    colsample_bytree=0.9,
    tree_method='hist',      
    device='cuda',           
    random_state=42,
    objective='multi:softprob',
    eval_metric='mlogloss'
)

# Lancement de l'apprentissage avec surveillance sur l'ensemble de test (eval_set)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100 # Affiche l'erreur tous les 100 arbres
)

# 4. ÉVALUATION DES PERFORMANCES

log(" Évaluation en cours...")
# Prédiction des types de drones sur le set de test
y_pred = model.predict(X_test)

# Inversion de l'encodage numérique pour retrouver les noms de drones originaux
y_test_orig = le.inverse_transform(y_test)
y_pred_orig = le.inverse_transform(y_pred)

# Calcul du score global d'exactitude (Accuracy)
acc = accuracy_score(y_test_orig, y_pred_orig)
# Génération du rapport de classification au format dictionnaire pour le JSON
report_dict = classification_report(y_test_orig, y_pred_orig, target_names=target_names, output_dict=True)
# Calcul de la matrice de confusion
cm = confusion_matrix(y_test_orig, y_pred_orig)

print(f"\n PRÉCISION FINALE (TYPE-ONLY) : {acc*100:.2f}%")
print("\n Rapport détaillé :")
print(classification_report(y_test_orig, y_pred_orig, target_names=target_names))

# 5. SAUVEGARDE POUR LE TABLEAU DE BORD

# Création d'un dictionnaire pour stocker toutes les statistiques finales
results_json = {
    "model_name": "XGBoost (Type Only - GPU)",
    "accuracy": float(acc),
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "target_names": target_names,
    "trained_at": datetime.now().isoformat()
}

# Définition des chemins de sauvegarde pour le JSON et le modèle binaire
json_path = os.path.join(SAVE_DIR, "xgboost_type_results.json")
pkl_path  = os.path.join(SAVE_DIR, "xgboost_type_model.pkl")

# Écriture du fichier JSON sur le disque
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=4)

# Sauvegarde binaire du modèle XGBoost pour réutilisation ultérieure
joblib.dump(model, pkl_path)

log(f" Succès ! Résultats sauvegardés dans {SAVE_DIR}")
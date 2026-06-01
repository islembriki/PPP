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

# Dossier où seront sauvegardés le modèle, l'encodeur et les statistiques JSON
SAVE_DIR = "./PPP/ml_trained_models_mode_included"
# Création du dossier s'il n'existe pas encore
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# Chemin relatif vers le fichier CSV de données fusionnées
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"


# 1. CHARGEMENT ET INGÉNIERIE AVANCÉE (HYPER-ENGINEERING)

log(" Démarrage de l'entraînement XGBoost (Type + Mode)...")
# Chargement du dataset avec Pandas
df = pd.read_csv(DATASET_PATH)
# Nettoyage : suppression des valeurs infinies et des lignes vides
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Application de l'ingénierie des caractéristiques comportementales...")
# 1. Caractéristiques de base (
# Logarithme de la variance pour stabiliser la distribution
df['Log_Var']   = np.log10(np.abs(df['Variance']) + 1e-9)
# Interaction entre le pic de puissance (PAPR) et la moyenne du segment
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
# Ratio de forme (Kurtosis divisé par Skewness)
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

# 2. Interactions comportementales complexes
# Ratio de puissance crête par rapport à la variance (stabilité du signal)
df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + 1e-9)
# Calcul de l'énergie totale du signal sur le segment
df['Signal_Energy']  = (df['Mean']**2) + df['Variance']
# Somme des asymétries de la distribution (Kurtosis + Skewness)
df['Shape_Factor']   = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])
# Moyenne élevée à la puissance 3 pour détecter les biais fins des capteurs
df['Mean_Cube']      = df['Mean']**3

# 2. CIBLE : 10 CLASSES (TYPE_MODE)

# Création de l'étiquette combinée 
df['Target'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)
# Sélection des caractéristiques (X) en retirant les colonnes d'origine et la cible
X = df.drop(['Label', 'Mode', 'Target'], axis=1)
# y contient la variable cible textuelle
y = df['Target']
# Liste triée des noms de classes uniques
target_names = sorted(df['Target'].unique())

# XGBoost nécessite des labels numériques (0, 1, 2...), on utilise LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

log(f"Forme du dataset: {X.shape} | Classes: {len(target_names)} → {target_names}")

# Division en ensembles d'entraînement (80%) et de test (20%) avec équilibrage des classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

log(f"Entraînement: {len(X_train)} lignes | Test: {len(X_test)} lignes")


# 3. ENTRAÎNEMENT XGBOOST

log(f" Entraînement de XGBoost sur {X.shape[1]} caractéristiques et {len(X_train)} échantillons...")

# Configuration du classifieur XGBoost :
# tree_method='hist' : méthode optimisée basée sur des histogrammes
# device='cuda' : tente d'utiliser le GPU NVIDIA pour accélérer les calculs
model = XGBClassifier(
    n_estimators=300,        # Nombre total d'arbres de décision
    max_depth=6,             # Profondeur maximale de chaque arbre
    learning_rate=0.1,       # Pas d'apprentissage pour la correction des erreurs
    subsample=0.8,           # Utilise 80% des données par arbre pour éviter le surapprentissage
    colsample_bytree=0.8,    # Utilise 80% des caractéristiques par arbre
    use_label_encoder=False, # Désactive l'ancien encodeur interne
    eval_metric='mlogloss',  # Utilise la log-loss multiclasse comme métrique d'évaluation
    tree_method='hist',      # Algorithme rapide de construction des arbres
    device='cuda',           # Exécution sur GPU si CUDA est configuré, sinon CPU
    n_jobs=-1,               # Utilise tous les cœurs CPU disponibles
    random_state=42,         # Fixe la graine pour la reproductibilité
    verbosity=1              # Affiche les messages d'avertissement et d'info
)

# Entraînement avec surveillance du set de test (eval_set)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50               # Affiche l'évolution du score tous les 50 arbres
)


# 4. ÉVALUATION DES PERFORMANCES

log(" Évaluation sur le test set complet...")
# Prédiction des classes sur les données de test
y_pred = model.predict(X_test)

# Reconversion des étiquettes numériques vers les noms originaux (ex: 0 -> "Bebop_M1")
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)

# Calcul du score global
acc    = accuracy_score(y_test_labels, y_pred_labels)
# Génération du rapport de classification au format dictionnaire pour le JSON
report = classification_report(y_test_labels, y_pred_labels, target_names=target_names, output_dict=True)
# Calcul de la matrice de confusion
cm     = confusion_matrix(y_test_labels, y_pred_labels)

print(f"\n PRÉCISION XGBOOST (ACCURACY): {acc*100:.2f}%")
print("\n Rapport de Classification:")
print(classification_report(y_test_labels, y_pred_labels, target_names=target_names))


# 5. SAUVEGARDE DU MODÈLE ET DES RÉSULTATS

# Préparation des statistiques et métriques pour le fichier JSON
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

# Définition des chemins de fichiers de sauvegarde
json_path = os.path.join(SAVE_DIR, "xgboost_mode.json")
pkl_path  = os.path.join(SAVE_DIR, "xgboost_mode.pkl")

# Sauvegarde du fichier JSON
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=4)

# Sauvegarde binaire du modèle XGBoost
joblib.dump(model, pkl_path)
# Sauvegarde de l'encodeur de labels (indispensable pour les prédictions futures)
joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder.pkl"))

log(f" Terminé ! Modèle sauvegardé dans {pkl_path}")
log(f"Statistiques sauvegardées dans {json_path}")
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

# Fonction pour afficher des messages avec l'heure (logs) dans la console
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# 1. CHEMINS DE FICHIERS 
# Chemin relatif vers le fichier CSV de données global
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"
# Dossier de sauvegarde pour les modèles de type seul (Background, Bebop, etc.)
SAVE_DIR = "./PPP/ml_trained_models_type_only"
# Création du dossier s'il n'existe pas encore
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# 2. CHARGEMENT ET INGÉNIERIE DES DONNÉES

log(" Démarrage de l'entraînement KNN Optimisé...")
# Chargement du fichier CSV avec Pandas
df = pd.read_csv(DATASET_PATH)
# Nettoyage : remplacement des valeurs infinies par NaN et suppression des lignes vides
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Application de l'ingénierie des caractéristiques (Logique 82%)...")
# Ajout de caractéristiques calculées pour mieux différencier les drones Bebop et AR Drone
# Calcul du logarithme de la variance
df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
# Produit entre le pic de puissance (PAPR) et la moyenne
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
# Ratio entre la forme (Kurtosis) et l'asymétrie (Skewness)
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

# Définition des 8 colonnes de caractéristiques finales utilisées pour l'entraînement
feature_cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Log_Var', 'PAPR_Mean', 'Kurt_Skew']
X = df[feature_cols] # Variables explicatives
y = df['Label']      # Variable cible (le type de drone)

# Noms des catégories pour l'affichage des résultats
target_names = ['Background', 'Bebop', 'AR_Drone', 'Phantom']

# 3. DÉCOUPAGE ET MISE À L'ÉCHELLE 

# Séparation des données : 80% entraînement, 20% test, avec équilibrage des classes (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialisation du scaler (indispensable pour KNN car il repose sur des calculs de distance)
scaler = StandardScaler()
# Apprentissage et application de la normalisation sur les données d'entraînement
X_train_scaled = scaler.fit_transform(X_train)
# Application de la même normalisation sur les données de test
X_test_scaled = scaler.transform(X_test)
log(" Ingénierie des données et normalisation terminées.")

# 4. ENTRAÎNEMENT DU KNN (k=52 avec poids par distance)

log(" Entraînement du KNN... (Cela prendra quelques minutes)")
# n_neighbors=52 : nombre de voisins consultés pour la décision
# weights='distance' : les voisins les plus proches ont plus d'influence sur le vote
# n_jobs=-1 : utilise tous les processeurs disponibles pour accélérer le calcul
knn = KNeighborsClassifier(n_neighbors=52, weights='distance', n_jobs=-1)
# Lancement de la phase d'apprentissage
knn.fit(X_train_scaled, y_train)

# 5. RÉSULTATS ET SAUVEGARDE

log(" Prédiction en cours...")
# Prédiction des types de drones sur le jeu de test
y_pred = knn.predict(X_test_scaled)
# Calcul du score de précision globale
acc = accuracy_score(y_test, y_pred)
# Génération du rapport de classification textuel
report = classification_report(y_test, y_pred, target_names=target_names)
# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred)

print("\n" + "="*40)
print(f" PRÉCISION FINALE DU KNN : {acc*100:.2f}%")
print("="*40)
print(report)

# --- SAUVEGARDE DU MODÈLE (PKL) ---
# Sauvegarde binaire du modèle KNN entraîné
joblib.dump(knn,    os.path.join(SAVE_DIR, "knn_model.pkl"))
# Sauvegarde du scaler pour pouvoir traiter les nouvelles données futures
joblib.dump(scaler, os.path.join(SAVE_DIR, "knn_scaler.pkl"))

# --- SAUVEGARDE DES RÉSULTATS (JSON) ---
# Préparation du dictionnaire de résultats pour export JSON
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
                                 output_dict=True          # Format dictionnaire pour le JSON
                             ),
    "confusion_matrix":      cm.tolist(),
    "dataset_shape": {
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "n_features":    int(X.shape[1])
    }
}

# Écriture du fichier JSON sur le disque
with open(os.path.join(SAVE_DIR, "knn_results.json"), 'w') as f:
    json.dump(results_json, f, indent=2)

log(" Tous les fichiers ont été sauvegardés dans ml_trained_models_type_only")
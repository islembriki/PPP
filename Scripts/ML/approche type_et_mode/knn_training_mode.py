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

# Fonction pour afficher des messages avec l'heure précise (log)
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# 0. CONFIGURATION DES CHEMINS 

# Dossier où seront sauvegardés les modèles entraînés (pkl et json)
SAVE_DIR = "./PPP/ml_trained_models_mode_included"
# Création du dossier s'il n'existe pas encore sur le disque
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# Chemin relatif vers le fichier de données global
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"


# 1. CHARGEMENT ET INGÉNIERIE DES DONNÉES

log(" Démarrage du KNN (Type + Mode)...")
# Lecture du fichier CSV avec Pandas
df = pd.read_csv(DATASET_PATH)
# Nettoyage : suppression des valeurs infinies et des lignes vides (NaN)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Application de l'ingénierie comportementale...")
# Création de nouvelles caractéristiques (features) pour aider le modèle KNN
# Passage de la Variance en échelle logarithmique
df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
# Produit entre le PAPR et la Moyenne
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
# Ratio entre le Kurtosis et la Skewness
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)
# Ratio entre le PAPR et la Variance
df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + 1e-9)
# Énergie du signal (somme du carré de la moyenne et de la variance)
df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
# Facteur de forme (somme des valeurs absolues de kurtosis et skewness)
df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])

# 2. CIBLE (TARGET): 10 CLASSES (DRONE + MODE)

# On crée une étiquette combinée, par exemple "Bebop_M1"
df['Target'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)
# X contient les variables explicatives (on retire les étiquettes)
X = df.drop(['Label', 'Mode', 'Target'], axis=1)
# y contient la variable à prédire (Target)
y = df['Target']
# Liste des noms de classes triée par ordre alphabétique
target_names = sorted(df['Target'].unique())

# Division des données : 80% pour l'entraînement et 20% pour le test
# On utilise stratify=y pour garder la même proportion de classes dans les deux sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. MISE À L'ÉCHELLE (OBLIGATOIRE POUR KNN)

log("Normalisation des données...")
# Le KNN calcule des distances, il faut donc que toutes les données soient sur la même échelle
scaler = StandardScaler()
# On calcule les paramètres sur le train et on les applique aux deux sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. ENTRAÎNEMENT (k=22 avec poids par distance)

log(" Entraînement du KNN (k=22, weights='distance')...")
# n_neighbors=22 définit le nombre de voisins consultés
# weights='distance' donne plus de poids aux voisins les plus proches
# n_jobs=-1 utilise tous les cœurs du processeur pour accélérer le calcul
knn = KNeighborsClassifier(n_neighbors=22, weights='distance', n_jobs=-1)
# Lancement de l'apprentissage
knn.fit(X_train_scaled, y_train)


# 5. ÉVALUATION DU MODÈLE

log("Prédiction en cours...")
# Prédiction sur les données de test que le modèle n'a jamais vues
y_pred = knn.predict(X_test_scaled)
# Calcul de l'exactitude (accuracy)
acc = accuracy_score(y_test, y_pred)
# Génération d'un rapport complet (précision, rappel, f1-score)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\n RÉSULTAT KNN (TYPE+MODE):")
print(f"Accuracy: {acc*100:.2f}%")
print("-" * 30)


# 6. SAUVEGARDE DES FICHIERS (PKL + JSON)

# 1. Sauvegarde binaire du modèle KNN pour une utilisation future
joblib.dump(knn, os.path.join(SAVE_DIR, "knn_mode.pkl"))
# Sauvegarde du scaler pour pouvoir normaliser les futures nouvelles données
joblib.dump(scaler, os.path.join(SAVE_DIR, "knn_scaler_mode.pkl"))

# 2. Préparation du dictionnaire de résultats pour le fichier JSON
results_json = {
    "model_name": "K-Nearest Neighbors (Type + Mode)",
    "accuracy": float(acc),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "target_names": target_names,
    "trained_at": datetime.now().isoformat()
}

# Écriture du fichier JSON de manière lisible (indentation de 4)
json_path = os.path.join(SAVE_DIR, "knn_mode.json")
with open(json_path, 'w') as f:
    json.dump(results_json, f, indent=4)

log(f" KNN Results saved to: {json_path}")
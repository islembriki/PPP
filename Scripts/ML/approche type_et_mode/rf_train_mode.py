import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# Fonction utilitaire pour afficher des messages horodatés dans la console
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# 0. CONFIGURATION DES CHEMINS RELATIFS

# Dossier où seront sauvegardés le modèle (.pkl) et les résultats (.json)
SAVE_DIR = "./PPP/ml_trained_models_mode_included"
# Création automatique du dossier s'il n'existe pas encore
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# Chemin relatif vers le fichier CSV contenant les données globales
DATASET_PATH = "./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv"

# 1. CHARGEMENT ET INGÉNIERIE AVANCÉE (HYPER-ENGINEERING)

log(" Démarrage de l'entraînement RF (Haute Précision)...")
# Chargement du dataset avec Pandas
df = pd.read_csv(DATASET_PATH)
# Nettoyage : suppression des valeurs infinies et des lignes contenant des données manquantes
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Application de l'ingénierie des caractéristiques comportementales...")
# 1. Caractéristiques de base ("Secret Sauce")
# Logarithme de la variance pour compresser l'échelle des données
df['Log_Var'] = np.log10(np.abs(df['Variance']) + 1e-9)
# Interaction entre le pic de puissance (PAPR) et la moyenne
df['PAPR_Mean'] = df['PAPR'] * df['Mean']
# Ratio de forme du signal (Kurtosis divisé par Skewness)
df['Kurt_Skew'] = df['Kurtosis'] / (df['Skewness'] + 1e-9)

# 2. NOUVELLES CARACTÉRISTIQUES : Interactions comportementales (pour séparer les modes de vol)
# Ratio entre puissance crête et stabilité (aide à séparer le vol stationnaire du vol actif)
df['PAPR_Var_Ratio'] = df['PAPR'] / (df['Variance'] + 1e-9)
# Indicateur d'énergie globale du segment
df['Signal_Energy'] = (df['Mean']**2) + df['Variance']
# Indicateur de la forme de la distribution (somme des asymétries)
df['Shape_Factor'] = np.abs(df['Kurtosis']) + np.abs(df['Skewness'])
# Moyenne au cube pour accentuer les micros-décalages des capteurs
df['Mean_Cube'] = df['Mean']**3


# 2. DÉFINITION DE LA CIBLE : 10 CLASSES

# On combine le type de drone et le mode de vol en une seule étiquette (ex: "Bebop_M1")
df['Target'] = df['Label'].astype(str) + "_M" + df['Mode'].astype(str)
# X contient les variables explicatives (on retire les colonnes d'origine et la cible)
X = df.drop(['Label', 'Mode', 'Target'], axis=1)
# y contient la variable cible à prédire
y = df['Target']
# Liste triée des noms de classes uniques
target_names = sorted(df['Target'].unique())

# Division des données en Train (80%) et Test (20%) avec stratification pour équilibrer les classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. ENTRAÎNEMENT HAUTE INTENSITÉ

log(f" Entraînement de l'Extreme Forest (500 arbres) sur {X.shape[1]} caractéristiques...")
# Configuration du Random Forest :
# n_estimators=500 : beaucoup d'arbres pour capturer les nuances des drones Parrot
# max_depth=40 : permet une croissance profonde pour plus de précision
# class_weight='balanced' : ajuste les poids si certaines classes sont moins représentées
# n_jobs=-1 : utilise toute la puissance du processeur (tous les cœurs)
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
# Lancement de l'apprentissage sur les données d'entraînement
rf.fit(X_train, y_train)


# 4. RÉSULTATS & GÉNÉRATION DU JSON

# Prédiction sur les données de test
y_pred = rf.predict(X_test)
# Calcul du score de précision globale
acc = accuracy_score(y_test, y_pred)
# Génération du rapport détaillé de classification (precision, recall, f1)
report = classification_report(y_test, y_pred, output_dict=True)

print(f"\n NOUVELLE PRÉCISION (ACCURACY): {acc*100:.2f}%")

# Préparation du dictionnaire de résultats pour le fichier JSON
results_json = {
    "model_name": "Random Forest (Optimized)",
    "accuracy": float(acc),
    "classification_report": report,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "target_names": target_names,
    "trained_at": datetime.now().isoformat()
}

# Sauvegarde des résultats au format JSON
with open(os.path.join(SAVE_DIR, "rf_mode.json"), 'w') as f:
    json.dump(results_json, f, indent=4)

# Sauvegarde binaire du modèle entraîné pour pouvoir le réutiliser sans réentraîner
joblib.dump(rf, os.path.join(SAVE_DIR, "rf_mode.pkl"))

log("Terminé ! Si la précision est toujours < 75%, cela prouve qu'un modèle CNN est nécessaire.")
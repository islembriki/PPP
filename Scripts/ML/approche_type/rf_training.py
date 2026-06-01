import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
from datetime import datetime

# Fonction pour afficher des messages horodatés dans la console
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# 1. CHARGEMENT DES DONNÉES

log("Chargement du dataset corrigé...")
# Utilisation d'un chemin relatif vers le fichier CSV global
df = pd.read_csv(r"./PPP/processed data/ML/FINAL_GLOBAL_DRONE_DATASET.csv")
# Nettoyage : suppression des valeurs infinies et des lignes vides (NaN)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

log("Application des transformations mathématiques...")
# Calcul du RMS (Root Mean Square) à partir de la racine de la variance
df['RMS']      = np.sqrt(np.abs(df['Variance']))
# Calcul de la valeur absolue de la moyenne
df['Abs_Mean'] = np.abs(df['Mean'])
# Transformation logarithmique du PAPR pour stabiliser la distribution
df['Log_PAPR'] = np.log10(df['PAPR'] + 1e-9)

# 2. PRÉPARATION DE LA CIBLE (TYPE UNIQUEMENT)

# X contient toutes les caractéristiques sauf les étiquettes de drone et de mode
X = df.drop(['Label', 'Mode'], axis=1)
# y contient uniquement le Label (type de drone : Background, Bebop, etc.)
y = df['Label']

# Liste des noms des catégories pour les rapports de performance
target_names = ['Background', 'Bebop', 'AR_Drone', 'Phantom']

# 3. DÉCOUPAGE ET NORMALISATION

# Séparation en 80% entraînement et 20% test avec stratification pour l'équilibre des classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialisation du scaler pour normaliser les données (moyenne 0, écart-type 1)
scaler = StandardScaler()
# Apprentissage et application sur le set d'entraînement
X_train_scaled = scaler.fit_transform(X_train)
# Application sur le set de test
X_test_scaled  = scaler.transform(X_test)
log(" Données mises à l'échelle")

# 4. ENTRAÎNEMENT DU MODÈLE
log(" Entraînement du Deep Forest...")

# Configuration du Random Forest :
# n_estimators=300 : nombre d'arbres de décision
# class_weight='balanced' : ajuste les poids pour les classes avec moins de données
# n_jobs=-1 : utilise toute la puissance du processeur
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Lancement de l'apprentissage
rf.fit(X_train_scaled, y_train)

# 5. ÉVALUATION DES PERFORMANCES

# Prédiction sur les données de test inconnues
y_pred = rf.predict(X_test_scaled)
# Calcul du score d'exactitude global
acc    = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print(" RÉSULTATS FINAUX : IDENTIFICATION TYPE DE DRONE")
print("="*60)
print(f"PRÉCISION (ACCURACY) : {acc*100:.2f}%")
print("\nRapport de Classification :")
# Affichage détaillé par classe (Background, Bebop, etc.)
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. SAUVEGARDE PKL (Modèle binaire)

# On s'assure que le dossier de destination existe
os.makedirs("./PPP/ml_trained_models_type_only", exist_ok=True)

# Sauvegarde binaire du modèle et du scaler (chemins relatifs)
joblib.dump(rf,     './PPP/ml_trained_models_type_only/drone_type_model_final.pkl')
joblib.dump(scaler, './PPP/ml_trained_models_type_only/scaler_final.pkl')
log(" Modèle et Scaler sauvegardés.")

# 7. SAUVEGARDE DES RÉSULTATS EN JSON

log("Sauvegarde des résultats dans rf_results.json ...")

# Génération du rapport de classification au format dictionnaire
report_dict = classification_report(
    y_test, y_pred,
    target_names=target_names,
    output_dict=True
)

# Conversion de la matrice de confusion en liste pour le format JSON
cm = confusion_matrix(y_test, y_pred).tolist()

# Association des colonnes avec leur score d'importance (quelles caractéristiques comptent le plus)
feature_importance = dict(
    zip(X.columns.tolist(), rf.feature_importances_.tolist())
)
# Tri des caractéristiques par importance décroissante
feature_importance_sorted = dict(
    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
)

# Création du dictionnaire final des résultats
results = {
    "trained_at":          datetime.now().isoformat(),
    "model":               "RandomForestClassifier",
    "n_estimators":        300,
    "class_weight":        "balanced",
    "test_size":           0.2,
    "random_state":          42,
    "accuracy":            round(acc * 100, 4),
    "target_names":        target_names,
    "classification_report": report_dict,
    "confusion_matrix":    cm,
    "feature_importances": feature_importance_sorted,
    "dataset_shape": {
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples":  int(len(X_test)),
        "n_features":    int(X.shape[1])
    }
}

# Écriture du fichier JSON final
with open('./PPP/ml_trained_models_type_only/rf_results.json', 'w') as f:
    json.dump(results, f, indent=2)

log(" rf_results.json sauvegardé — prêt pour le tableau de bord !")
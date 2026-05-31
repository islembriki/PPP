import joblib
import pandas as pd

# Charger le modèle
model = joblib.load(r"C:\Users\garba\Desktop\PPP FINAL\PPP\models\random_forest.pkl")

print("--- CONTENU DU RANDOM FOREST ---")
# Voir l'importance des caractéristiques
importances = model.feature_importances_
features = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR']

for f, imp in zip(features, importances):
    print(f"L'indice {f} aide l'IA à hauteur de : {imp*100:.2f}%")
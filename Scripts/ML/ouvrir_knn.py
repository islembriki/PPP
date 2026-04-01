import joblib

# 1. Charger le modèle
knn = joblib.load(r"C:\Users\garba\Desktop\PPP\models\knn_model.pkl")

print("--- INSPECTION DU KNN ---")
# Voir combien de voisins il regarde pour décider
print(f"Nombre de voisins consultés (K) : {knn.n_neighbors}")

# Voir la méthode de calcul de distance (souvent 'minkowski' ou 'euclidean')
print(f"Méthode de calcul de distance : {knn.effective_metric_}")

# Vérifier qu'il a bien les données en mémoire
print(f"Nombre de points stockés en mémoire : {knn._fit_X.shape[0]}")
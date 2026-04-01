import joblib

# 1. Charger le modèle
svm = joblib.load(r"C:\Users\garba\Desktop\PPP\models\svm_model.pkl")

print("--- INSPECTION DU SVM ---")
# Voir combien de "points clés" (vecteurs de support) il a retenu par drone
print(f"Nombre de vecteurs de support par classe : {svm.n_support_}")

# Voir les paramètres mathématiques qu'il utilise
print(f"Noyau utilisé (Kernel) : {svm.kernel}")
print(f"Paramètre de régularisation (C) : {svm.C}")
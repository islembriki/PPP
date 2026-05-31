import joblib

# Charger le scaler
scaler = joblib.load(r"C:\Users\garba\Desktop\PPP FINAL\PPP\models\scaler.pkl")

print("--- CONTENU DU SCALER ---")
print("Moyennes apprises pour chaque colonne (Mean, Var, Kurt, Skew, PAPR) :")
print(scaler.mean_)
print("\nÉcarts-types appris :")
print(scaler.scale_)
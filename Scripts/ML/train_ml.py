import pandas as pd
import numpy as np
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- FONCTION PRINCIPALE (COMBINE TERMINAL + GUI) ---
def demarrer_ia():
    # 1. Configuration des chemins
    path_dataset = r"C:\Users\garba\Desktop\PPP\processed data\ML\GLOBAL_DRONE_DATASET.csv"
    path_models = r"C:\Users\garba\Desktop\PPP\models"

    if not os.path.exists(path_dataset):
        print(f"ERREUR : Le fichier {path_dataset} est introuvable.")
        messagebox.showerror("Erreur", "Fichier dataset introuvable !")
        return

    try:
        # --- PHASE 1 : CHARGEMENT ---
        print("\n" + "="*40)
        print("Étape 1 : Chargement des données...")
        label_statut.config(text="Statut : Chargement des données...", foreground="blue")
        root.update()

        df = pd.read_csv(path_dataset)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        print(f"Dataset chargé : {df.shape[0]} exemples trouvés.")
        print("Répartition des drones :")
        print(df['Label'].value_counts())
        
        # Séparation X et y
        X = df.drop('Label', axis=1)
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisation
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        if not os.path.exists(path_models): os.makedirs(path_models)
        joblib.dump(scaler, os.path.join(path_models, "scaler.pkl"))
        print("Scaler sauvegardé !")

        # --- PHASE 2 : SVM ---
        print("\n--- Entraînement du modèle SVM ---")
        label_statut.config(text="Statut : Entraînement SVM en cours...")
        root.update()
        
        svm_model = SVC(kernel='rbf', C=1.0)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        
        print(f"Précision du SVM : {acc_svm*100:.2f}%")
        val_svm.config(text=f"{acc_svm*100:.2f} %") # Update GUI
        joblib.dump(svm_model, os.path.join(path_models, "svm_model.pkl"))

        # --- PHASE 3 : RANDOM FOREST ---
        print("\n--- Entraînement du modèle Random Forest ---")
        label_statut.config(text="Statut : Entraînement Random Forest en cours...")
        root.update()
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        
        print(f"Précision du Random Forest : {acc_rf*100:.2f}%")
        val_rf.config(text=f"{acc_rf*100:.2f} %") # Update GUI
        joblib.dump(rf_model, os.path.join(path_models, "random_forest_model.pkl"))

        # --- PHASE 4 : KNN ---
        print("\n--- Entraînement du modèle KNN ---")
        label_statut.config(text="Statut : Entraînement KNN en cours...")
        root.update()
        
        knn_model = KNeighborsClassifier(n_neighbors=5) 
        knn_model.fit(X_train, y_train)
        y_pred_knn = knn_model.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        
        print(f"Précision du KNN : {acc_knn*100:.2f}%")
        val_knn.config(text=f"{acc_knn*100:.2f} %") # Update GUI
        joblib.dump(knn_model, os.path.join(path_models, "knn_model.pkl"))

        # --- PHASE 5 : RAPPORT FINAL ---
        print("\n" + "="*40)
        print("RÉSULTATS FINAUX DANS LE TERMINAL :")
        rapport = classification_report(y_test, y_pred_rf)
        print(rapport)
        
        # Update GUI Text Area
        text_rapport.delete('1.0', tk.END)
        text_rapport.insert(tk.END, rapport)
        
        label_statut.config(text="Statut : Entraînement Terminé !", foreground="green")
        print(f"\nTous les modèles ont été sauvegardés dans : {path_models}")
        messagebox.showinfo("Succès", "Entraînement terminé et affiché !")

    except Exception as e:
        print(f"ERREUR : {str(e)}")
        messagebox.showerror("Erreur", f"Erreur : {str(e)}")

# --- CRÉATION DE LA FENÊTRE (GUI) ---
root = tk.Tk()
root.title("Interface IA - Drone Classification")
root.geometry("600x650")
root.configure(padx=20, pady=20)

# Titre
tk.Label(root, text="Dashboard Apprentissage IA", font=("Arial", 16, "bold")).pack(pady=10)

# Bouton
btn_go = ttk.Button(root, text="LANCER L'ENTRAÎNEMENT ", command=demarrer_ia)
btn_go.pack(pady=10, fill=tk.X)

# Statut
label_statut = tk.Label(root, text="Statut : Prêt", font=("Arial", 10, "italic"))
label_statut.pack(pady=5)

# Frame pour les scores
frame_res = tk.LabelFrame(root, text=" Précision des Modèles ", padx=10, pady=10)
frame_res.pack(pady=10, fill=tk.X)

tk.Label(frame_res, text="SVM Accuracy :").grid(row=0, column=0, sticky="w")
val_svm = tk.Label(frame_res, text="-- %", font=("Arial", 10, "bold"), fg="blue")
val_svm.grid(row=0, column=1, padx=20)

tk.Label(frame_res, text="Random Forest Accuracy :").grid(row=1, column=0, sticky="w")
val_rf = tk.Label(frame_res, text="-- %", font=("Arial", 10, "bold"), fg="green")
val_rf.grid(row=1, column=1, padx=20)

tk.Label(frame_res, text="KNN Accuracy :").grid(row=2, column=0, sticky="w")
val_knn = tk.Label(frame_res, text="-- %", font=("Arial", 10, "bold"), fg="purple")
val_knn.grid(row=2, column=1, padx=20)

# Zone de texte pour le rapport
tk.Label(root, text="Rapport détaillé (Random Forest) :", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10,0))
text_rapport = tk.Text(root, height=12, font=("Courier New", 9))
text_rapport.pack(fill=tk.BOTH, pady=5)

root.mainloop()
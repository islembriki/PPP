import pandas as pd
import numpy as np
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
# Data Science & ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Visualisation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# ← NEW: log function
def log(msg):                                                    # ← NEW
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)  # ← NEW
# --- FONCTION DE TRAITEMENT ---
def demarrer_ia():
    path_dataset = r"C:\Users\HP\Desktop\PPP\processed data\ML\GLOBAL_DRONE_DATASET.csv"
    path_models  = r"C:\Users\HP\Desktop\PPP\ml_trained_models"

    if not os.path.exists(path_dataset):
        messagebox.showerror("Erreur", "Fichier dataset introuvable !")
        return

    try:
        label_statut.config(text="Statut : Chargement et préparation...", foreground="blue")
        root.update()

        # 1. Chargement et Nettoyage
        df = pd.read_csv(path_dataset)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        log(f"✓ CSV loaded: {df.shape[0]} rows") 
        # --- CRÉATION DE LA CIBLE COMBINÉE (Type + Mode) ---
        df['Target'] = df['Label'].astype(str) + " (M" + df['Mode'].astype(str) + ")"

        X = df.drop(['Label', 'Mode', 'Target'], axis=1)
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        log(f"✓ Split done — Train: {len(X_train)}, Test: {len(X_test)}")  # ← NEW
        # 2. Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        if not os.path.exists(path_models):
            os.makedirs(path_models)

        # ✅ SAVE SCALER
        joblib.dump(scaler, os.path.join(path_models, "scaler.pkl"))
        print("✓ Scaler saved")

        # --- ENTRAÎNEMENT ---

        # SVM
        log("⏳ Starting SVM training... (this is the long one)")  # ← NEW
        label_statut.config(text="Statut : Entraînement SVM (peut prendre longtemps)...", foreground="orange")
        root.update()
        svm = SVC(kernel='rbf', C=1.0, verbose=True) # ← NEW: verbose=True for progress
        svm.fit(X_train_scaled, y_train)
        acc_svm = accuracy_score(y_test, svm.predict(X_test_scaled))
        val_svm.config(text=f"{acc_svm*100:.2f} %")
        # ✅ SAVE SVM
        joblib.dump(svm, os.path.join(path_models, "svm.pkl"))
        print("✓ SVM saved")
        log(f"✓ SVM done! Accuracy: {acc_svm*100:.2f}%") 
        # Random Forest
        log("⏳ Starting Random Forest training...")  
        label_statut.config(text="Statut : Entraînement Random Forest...", foreground="orange")
        root.update()
        rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2, n_jobs=-1) 
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        val_rf.config(text=f"{acc_rf*100:.2f} %")
        # ✅ SAVE RANDOM FOREST
        joblib.dump(rf, os.path.join(path_models, "random_forest.pkl"))
        print("✓ Random Forest saved")
        log(f"✓ Random Forest done! Accuracy: {acc_rf*100:.2f}%")
        
        # KNN
        log("⏳ Starting KNN training...") 
        label_statut.config(text="Statut : Entraînement KNN...", foreground="orange")
        root.update()
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1) 
        knn.fit(X_train_scaled, y_train)
        acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
        val_knn.config(text=f"{acc_knn*100:.2f} %")
        # ✅ SAVE KNN
        joblib.dump(knn, os.path.join(path_models, "knn.pkl"))
        print("✓ KNN saved")
        log(f"✓ KNN done! Accuracy: {acc_knn*100:.2f}%")        # ← NEW
        log("🎉 ALL MODELS SAVED SUCCESSFULLY")  

        # --- GRAPHIQUE DE COMPARAISON DANS L'INTERFACE ---
        tracer_comparaison(acc_svm, acc_rf, acc_knn)

        # Rapport Final (based on Random Forest)
        rapport = classification_report(y_test, y_pred_rf)
        text_rapport.delete('1.0', tk.END)
        text_rapport.insert(tk.END, rapport)

        label_statut.config(text="Statut : Terminé ! Modèles sauvegardés.", foreground="green")
        messagebox.showinfo("Succès", f"Modèles sauvegardés dans :\n{path_models}\n\n"
                                       f"svm.pkl\nrandom_forest.pkl\nknn.pkl\nscaler.pkl")

    except Exception as e:
        log(f"✗ ERROR: {str(e)}") 
        messagebox.showerror("Erreur", f"Détails : {str(e)}")


def tracer_comparaison(s, r, k):
    for widget in frame_graph.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(4, 3))
    algos  = ['SVM', 'R.Forest', 'KNN']
    scores = [s*100, r*100, k*100]
    colors = ['#3498db', '#2ecc71', '#9b59b6']

    ax.bar(algos, scores, color=colors)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Précision (%)')
    ax.set_title('Comparaison des Performances')

    canvas = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas.draw()
    canvas.get_tk_widget().pack()


# --- INTERFACE GRAPHIQUE (GUI) ---
root = tk.Tk()
root.title("Drone Intel - Classification Type & Mode")
root.geometry("900x750")

# Colonne Gauche (Contrôles et Scores)
frame_left = tk.Frame(root, padx=20, pady=20)
frame_left.pack(side=tk.LEFT, fill=tk.Y)

tk.Label(frame_left, text="Dashboard Apprentissage", font=("Arial", 14, "bold")).pack(pady=10)

btn_go = ttk.Button(frame_left, text="LANCER L'ANALYSE COMPLÈTE", command=demarrer_ia)
btn_go.pack(pady=10, fill=tk.X)

label_statut = tk.Label(frame_left, text="Statut : Prêt", font=("Arial", 9, "italic"))
label_statut.pack()

# Affichage des scores
frame_scores = tk.LabelFrame(frame_left, text=" Précision (Acc.) ", padx=10, pady=10)
frame_scores.pack(pady=20, fill=tk.X)

tk.Label(frame_scores, text="SVM :").grid(row=0, column=0, sticky="w")
val_svm = tk.Label(frame_scores, text="-- %", font=("Arial", 10, "bold"), fg="blue")
val_svm.grid(row=0, column=1, padx=10)

tk.Label(frame_scores, text="Random Forest :").grid(row=1, column=0, sticky="w")
val_rf = tk.Label(frame_scores, text="-- %", font=("Arial", 10, "bold"), fg="green")
val_rf.grid(row=1, column=1, padx=10)

tk.Label(frame_scores, text="KNN :").grid(row=2, column=0, sticky="w")
val_knn = tk.Label(frame_scores, text="-- %", font=("Arial", 10, "bold"), fg="purple")
val_knn.grid(row=2, column=1, padx=10)

# Zone Rapport
tk.Label(frame_left, text="Rapport détaillé :", font=("Arial", 10, "bold")).pack(anchor="w")
text_rapport = tk.Text(frame_left, height=15, width=45, font=("Courier New", 8))
text_rapport.pack(pady=5)

# Colonne Droite (Graphiques)
frame_right = tk.Frame(root, padx=20, pady=20)
frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

frame_graph = tk.Frame(frame_right, bg="white", relief="sunken", bd=1)
frame_graph.pack(expand=True, fill=tk.BOTH)
tk.Label(frame_graph, text="Le graphique de comparaison s'affichera ici").pack(expand=True)

root.mainloop()
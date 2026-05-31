import pandas as pd
import numpy as np
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
import threading  # <--- INDISPENSABLE POUR NE PLUS FREEZER

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

# --- FONCTION DE TRAITEMENT (ML) ---
def calcul_ia_process():
    path_dataset = r"C:\Users\garba\Desktop\PPP FINAL\PPP\processed data\ML\GLOBAL_DRONE_DATASET.csv"
    path_models = r"C:\Users\garba\Desktop\PPP FINAL\PPP\models"

    try:
        label_statut.config(text="Statut : Lecture du fichier CSV...", foreground="blue")
        
        
        df = pd.read_csv(path_dataset)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df['Target'] = df['Label'].astype(str) + " (M" + df['Mode'].astype(str) + ")"

        X = df.drop(['Label', 'Mode', 'Target'], axis=1)
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        if not os.path.exists(path_models): os.makedirs(path_models)
        joblib.dump(scaler, os.path.join(path_models, "scaler.pkl"))

        # SVM
        label_statut.config(text="Statut : Entraînement SVM (Long)...", foreground="orange")
     
        svm = SVC(kernel='rbf', C=1.0)
        svm.fit(X_train_scaled, y_train)
        acc_svm = accuracy_score(y_test, svm.predict(X_test_scaled))
        joblib.dump(svm, os.path.join(path_models, "svm.pkl"))

        # Random Forest
        label_statut.config(text="Statut : Entraînement Random Forest...", foreground="orange")
     
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict(X_test_scaled)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        joblib.dump(rf, os.path.join(path_models, "random_forest.pkl"))

        # KNN
        label_statut.config(text="Statut : Entraînement KNN...", foreground="orange")
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
        joblib.dump(knn, os.path.join(path_models, "knn.pkl"))

        # UI Updates (Final)
        val_svm.config(text=f"{acc_svm*100:.2f} %")
        val_rf.config(text=f"{acc_rf*100:.2f} %")
        val_knn.config(text=f"{acc_knn*100:.2f} %")
        tracer_comparaison(acc_svm, acc_rf, acc_knn)
        
        rapport = classification_report(y_test, y_pred_rf)
        text_rapport.delete('1.0', tk.END)
        text_rapport.insert(tk.END, rapport)

        label_statut.config(text="Statut : Entraînement terminé !", foreground="green")
        btn_go.config(state=tk.NORMAL)
        btn_load.config(state=tk.NORMAL)
        messagebox.showinfo("Succès", "Modèles sauvegardés avec succès !")

    except Exception as e:
        messagebox.showerror("Erreur", str(e))
        btn_go.config(state=tk.NORMAL)

def demarrer_ia():
    # On désactive les boutons pour éviter de cliquer deux fois
    btn_go.config(state=tk.DISABLED)
    btn_load.config(state=tk.DISABLED)
    # On lance le calcul dans un Thread séparé
    threading.Thread(target=calcul_ia_process, daemon=True).start()

# --- FONCTION DE CHARGEMENT (VIA THREAD) ---
def charger_ia_process():
    path_dataset = r"C:\Users\garba\Desktop\PPP FINAL\PPP\processed data\ML\GLOBAL_DRONE_DATASET.csv"
    path_models  = r"C:\Users\garba\Desktop\PPP FINAL\PPP\models"

    try:
        label_statut.config(text="Statut : Chargement des fichiers...", foreground="blue")
     
        
        df = pd.read_csv(path_dataset)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df['Target'] = df['Label'].astype(str) + " (M" + df['Mode'].astype(str) + ")"
        X = df.drop(['Label', 'Mode', 'Target'], axis=1)
        y = df['Target']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = joblib.load(os.path.join(path_models, "scaler.pkl"))
        X_test_scaled = scaler.transform(X_test)

        svm = joblib.load(os.path.join(path_models, "svm.pkl"))
        acc_svm = accuracy_score(y_test, svm.predict(X_test_scaled))
        
        rf = joblib.load(os.path.join(path_models, "random_forest.pkl"))
        y_pred_rf = rf.predict(X_test_scaled)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        
        knn = joblib.load(os.path.join(path_models, "knn.pkl"))
        acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))

        # Update UI
        val_svm.config(text=f"{acc_svm*100:.2f} %")
        val_rf.config(text=f"{acc_rf*100:.2f} %")
        val_knn.config(text=f"{acc_knn*100:.2f} %")
        tracer_comparaison(acc_svm, acc_rf, acc_knn)
        text_rapport.delete('1.0', tk.END)
        text_rapport.insert(tk.END, classification_report(y_test, y_pred_rf))

        label_statut.config(text="Statut : Modèles chargés !", foreground="green")
        btn_go.config(state=tk.NORMAL)
        btn_load.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Erreur", str(e))
        btn_load.config(state=tk.NORMAL)

def charger_ia_existante():
    btn_go.config(state=tk.DISABLED)
    btn_load.config(state=tk.DISABLED)
    threading.Thread(target=charger_ia_process, daemon=True).start()

def tracer_comparaison(s, r, k):
    for widget in frame_graph.winfo_children(): widget.destroy()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['SVM', 'R.Forest', 'KNN'], [s*100, r*100, k*100], color=['#3498db', '#2ecc71', '#9b59b6'])
    ax.set_ylim(0, 110)
    ax.set_title('Comparaison des Performances (%)')
    canvas = FigureCanvasTkAgg(fig, master=frame_graph)
    canvas.draw()
    canvas.get_tk_widget().pack()

# --- GUI ---
root = tk.Tk()
root.title("Drone Intel - Dashboard")
root.geometry("900x750")

frame_left = tk.Frame(root, padx=20, pady=20)
frame_left.pack(side=tk.LEFT, fill=tk.Y)

tk.Label(frame_left, text="Dashboard Apprentissage", font=("Arial", 14, "bold")).pack(pady=10)

btn_go = ttk.Button(frame_left, text="LANCER L'ANALYSE COMPLÈTE", command=demarrer_ia)
btn_go.pack(pady=10, fill=tk.X)
btn_load = ttk.Button(frame_left, text="CHARGER MODÈLES EXISTANTS", command=charger_ia_existante)
btn_load.pack(pady=5, fill=tk.X)

label_statut = tk.Label(frame_left, text="Statut : Prêt", font=("Arial", 9, "italic"))
label_statut.pack()

frame_scores = tk.LabelFrame(frame_left, text=" Précision (Acc.) ", padx=10, pady=10)
frame_scores.pack(pady=20, fill=tk.X)
tk.Label(frame_scores, text="SVM :").grid(row=0, column=0, sticky="w")
val_svm = tk.Label(frame_scores, text="-- %", font=("Arial", 10, "bold"), fg="blue"); val_svm.grid(row=0, column=1, padx=10)
tk.Label(frame_scores, text="Random Forest :").grid(row=1, column=0, sticky="w")
val_rf = tk.Label(frame_scores, text="-- %", font=("Arial", 10, "bold"), fg="green"); val_rf.grid(row=1, column=1, padx=10)
tk.Label(frame_scores, text="KNN :").grid(row=2, column=0, sticky="w")
val_knn = tk.Label(frame_scores, text="-- %", font=("Arial", 10, "bold"), fg="purple"); val_knn.grid(row=2, column=1, padx=10)

text_rapport = tk.Text(frame_left, height=15, width=45, font=("Courier New", 8))
text_rapport.pack(pady=5)

frame_right = tk.Frame(root, padx=20, pady=20)
frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
frame_graph = tk.Frame(frame_right, bg="white", relief="sunken", bd=1)
frame_graph.pack(expand=True, fill=tk.BOTH)

root.mainloop()
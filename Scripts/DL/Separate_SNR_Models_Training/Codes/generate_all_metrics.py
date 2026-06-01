import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from model import DroneCNN
from data_manager import DroneRFDataset, get_smart_splits

# 1. CONFIGURATION
DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERTS_DIR = os.path.join(BASE_DIR, "..", "Experts")

experts_info = [
    {"snr": 30, "file": "expert_30dB_13classes.pth"},
    {"snr": 10, "file": "expert_10dB_13classes.pth"},
    {"snr": 0,  "file": "expert_0dB_13classes.pth"},
    {"snr": -10,"file": "expert_-10dB_13classes.pth"}
]

FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Background\Background",
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\AR_Drone\AR_Drone",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom\Phantom"
]

CLASS_NAMES = ["Background", "Bebop M1", "Bebop M2", "Bebop M3", "Bebop M4", 
               "AR M1", "AR M2", "AR M3", "AR M4", "Phant M1", "Phant M2", "Phant M3", "Phant M4"]

def generate_global_report():
    summary_results = []
    
    print("Démarrage du calcul des métriques pour tous les experts...")

    for info in experts_info:
        snr = info["snr"]
        model_path = os.path.join(EXPERTS_DIR, info["file"])
        
        print(f"\nAnalyse Expert {snr}dB...")
        
        # Préparation des données spécifiques au SNR
        full_ds = DroneRFDataset(FOLDERS, target_snr=snr)
        _, val_ds = get_smart_splits(full_ds)
        loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        # Chargement modèle
        model = DroneCNN(nb_classes=13)
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE).eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                outputs = model(imgs.to(DEVICE))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(lbls.numpy())

        # Calcul du rapport (uniquement sur les classes présentes pour éviter l'erreur)
        present_labels = np.unique(all_labels)
        present_names = [CLASS_NAMES[i] for i in present_labels]
        
        report = classification_report(all_labels, all_preds, 
                                       labels=present_labels, 
                                       target_names=present_names, 
                                       output_dict=True)
        
        # 1. Sauvegarde du rapport détaillé en CSV
        df_detailed = pd.DataFrame(report).transpose()
        df_detailed.to_csv(f"metrics_detailed_{snr}dB.csv")
        
        # 2. Ajout au résumé global (Accuracy et F1-score moyen)
        summary_results.append({
            "SNR (dB)": snr,
            "Accuracy (%)": round(report["accuracy"] * 100, 2),
            "F1-Score Moyen": round(report["macro avg"]["f1-score"], 3),
            "Précision Moyenne": round(report["macro avg"]["precision"], 3),
            "Rappel Moyen": round(report["macro avg"]["recall"], 3)
        })

    # 3. Création du tableau de synthèse global
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv("synthese_performances_SNR_DL.csv", index=False)
    
    print("\n" + "="*50)
    print("✅ RAPPORTS GÉNÉRÉS AVEC SUCCÈS !")
    print("Fichiers créés :")
    print("- synthese_performances_SNR_DL.csv (Tableau global)")
    print("- metrics_detailed_XXdB.csv (Détails par classe)")
    print("="*50)
    print(df_summary)

if __name__ == "__main__":
    generate_global_report()
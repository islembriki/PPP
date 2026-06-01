import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# Imports locaux
from model import DroneCNN
from data_manager import DroneRFDataset, get_smart_splits

# ==========================================
# 1. CONFIGURATION AUTOMATIQUE
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERTS_DIR = os.path.join(CURRENT_DIR, "..", "Experts")
DEVICE = torch.device("cpu") 

FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Background\Background",
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\AR_Drone\AR_Drone",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom\Phantom"
]

CLASS_NAMES = [
    "Background",
    "Bebop M1", "Bebop M2", "Bebop M3", "Bebop M4",
    "AR M1",    "AR M2",    "AR M3",    "AR M4",
    "Phant M1", "Phant M2", "Phant M3", "Phant M4"
]

# Liste des SNR à générer
SNR_LIST = [10, 0, -10]

def generate_all_heatmaps():
    for snr in SNR_LIST:
        model_file = f"expert_{snr}dB_13classes.pth"
        model_path = os.path.join(EXPERTS_DIR, model_file)
        
        print(f"\n🔄 TRAITEMENT EXPERT {snr}dB...")
        
        if not os.path.exists(model_path):
            print(f"❌ Erreur : {model_file} introuvable dans le dossier Experts.")
            continue

        # --- Préparation des données spécifiques au SNR ---
        full_ds = DroneRFDataset(FOLDERS, target_snr=snr)
        _, val_ds = get_smart_splits(full_ds)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        # --- Chargement du modèle ---
        model = DroneCNN(nb_classes=13)
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE).eval()

        all_preds, all_labels = [], []

        print(f"🔍 Calcul des prédictions pour {snr}dB...")
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(DEVICE))
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(labels.numpy())

        # --- Création de la matrice ---
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(14, 11))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    annot_kws={"size": 9, "weight": "bold"})

        plt.title(f"MATRICE DE CONFUSION : EXPERT {snr}dB (13 CLASSES BUI)", 
                  fontsize=16, fontweight='bold', pad=20, color="#022b55")
        plt.xlabel("Classes Prédites", fontsize=14, fontweight='bold')
        plt.ylabel("Classes Réelles", fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Sauvegarde avec nom dynamique
        output_name = f"confusion_matrix_{snr}dB.png"
        plt.savefig(output_name, dpi=300)
        print(f"✅ Image enregistrée : {output_name}")
        
        # Rapport de texte (Correction du crash)
        present_labels = np.unique(np.concatenate([all_labels, all_preds]))
        present_names = [CLASS_NAMES[i] for i in present_labels]
        print(classification_report(all_labels, all_preds, labels=present_labels, target_names=present_names))
        
        plt.close() # On ferme pour ne pas saturer la RAM

    print("\n✨ Terminé ! Toutes les matrices ont été générées.")

if __name__ == "__main__":
    generate_all_heatmaps()
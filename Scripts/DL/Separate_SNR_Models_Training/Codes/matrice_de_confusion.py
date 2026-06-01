import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# Imports locaux (assure-toi d'être dans le dossier 'Codes' pour lancer le script)
from model import DroneCNN
from data_manager import DroneRFDataset, get_smart_splits

# ==========================================
# 1. CONFIGURATION AUTOMATIQUE DES CHEMINS
# ==========================================
# Récupère le dossier actuel (Codes)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemin vers le dossier Experts (un niveau au-dessus)
EXPERTS_DIR = os.path.join(CURRENT_DIR, "..", "Experts")
MODEL_FILE = "expert_30dB_13classes.pth"
MODEL_PATH = os.path.join(EXPERTS_DIR, MODEL_FILE)

DEVICE = torch.device("cpu") # CPU pour la stabilité de l'affichage

# Chemins vers les données spectrogrammes
FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Background\Background",
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\AR_Drone\AR_Drone",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom\Phantom"
]

# Noms pour les axes du graphique
CLASS_NAMES = [
    "Background",
    "Bebop M1", "Bebop M2", "Bebop M3", "Bebop M4",
    "AR M1",    "AR M2",    "AR M3",    "AR M4",
    "Phant M1", "Phant M2", "Phant M3", "Phant M4"
]

def generate_heatmap():
    print(f"Recherche du modèle dans : {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERREUR : Le fichier {MODEL_FILE} est introuvable dans le dossier Experts.")
        return

    # --- Préparation des données ---
    print("⏳ Chargement des données de test (SNR 30dB)...")
    full_ds = DroneRFDataset(FOLDERS, target_snr=30)
    _, val_ds = get_smart_splits(full_ds)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # --- Chargement du modèle ---
    model = DroneCNN(nb_classes=13)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    print("Calcul des prédictions...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # --- Création de la matrice ---
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    # Création de la Heatmap avec Seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)

    plt.title("MATRICE DE CONFUSION : EXPERT 30dB (13 CLASSES BUI)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Classes Prédites", fontsize=14, fontweight='bold')
    plt.ylabel("Classes Réelles", fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Sauvegarde de l'image
    output_image = "confusion_matrix_30dB.png"
    plt.savefig(output_image, dpi=300)
    print(f"\nSuccès ! Image enregistrée sous : {output_image}")
    
    # Affichage du rapport textuel pour vérification
    print("\n--- RAPPORT DE CLASSIFICATION ---")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    plt.show()

if __name__ == "__main__":
    generate_heatmap()
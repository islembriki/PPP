import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from model import DroneCNN
from data_manager import DroneRFDataset, get_smart_splits

# ==========================================
# 1. CONFIGURATION ET COULEURS FIXES
# ==========================================
DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERTS_DIR = os.path.join(BASE_DIR, "..", "Experts")

# COULEURS FIXES : Garanti que chaque drone garde sa couleur sur tout l'interface
COLOR_MAP = {
    "Background": "#26a5d8", # Bleu nuit / Gris
    "Bebop":      "#f039e1", # Rouge
    "AR Drone":   "#c8d243", # Orange
    "Phantom":    "#18C15E"  # Vert
}

# Info experts avec couleurs sémantiques pour les titres (Vert -> Rouge)
experts_info = [
    {"snr": 30, "file": "expert_30dB_13classes.pth", "ui_color": "#40ac44"},
    {"snr": 10, "file": "expert_10dB_13classes.pth", "ui_color": "#4de9d9"},
    {"snr": 0,  "file": "expert_0dB_13classes.pth",  "ui_color": "#a6aa2f"},
    {"snr": -10,"file": "expert_-10dB_13classes.pth","ui_color": "#e13f4c"}
]

FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Background\Background",
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\AR_Drone\AR_Drone",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom\Phantom"
]

def get_features(model, loader):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            x = model.pool(torch.relu(model.conv1(imgs)))
            x = model.pool(torch.relu(model.conv2(x)))
            x = model.pool(torch.relu(model.conv3(x)))
            x = x.view(-1, 128 * 16 * 16)
            feat = torch.relu(model.fc1(x))
            features.append(feat.numpy())
            
            for l in lbls.numpy():
                if l == 0: labels.append("Background")
                elif 1 <= l <= 4: labels.append("Bebop")
                elif 5 <= l <= 8: labels.append("AR Drone")
                else: labels.append("Phantom")
    return np.concatenate(features), labels

def run_multi_tsne():
    # Création de l'interface
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.patch.set_facecolor('#f8f9fa') # Fond de page gris très clair
    
    fig.suptitle("ANALYSE DE L'ESPACE LATENT (t-SNE) : SÉPARATION DES SIGNATURES\nComparaison de la résilience des modèles selon le niveau de bruit (SNR)", 
                 fontsize=28, fontweight='bold', color='#2c3e50', y=0.97)
    
    axes = axes.flatten()

    for i, info in enumerate(experts_info):
        ax = axes[i]
        model_path = os.path.join(EXPERTS_DIR, info["file"])
        
        print(f"Calcul t-SNE pour l'expert {info['snr']}dB...")
        
        # Données
        full_ds = DroneRFDataset(FOLDERS, target_snr=info["snr"])
        _, val_ds = get_smart_splits(full_ds)
        indices = np.random.choice(len(val_ds), 800, replace=False)
        loader = DataLoader(torch.utils.data.Subset(val_ds, indices), batch_size=32, shuffle=False)

        # Modèle
        model = DroneCNN(nb_classes=13)
        model.load_state_dict(torch.load(model_path))
        model.to(DEVICE).eval()

        # Features et TSNE
        feat, lbls = get_features(model, loader)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        res = tsne.fit_transform(feat)

        # Plot avec palette FIXE
        sns.scatterplot(x=res[:,0], y=res[:,1], hue=lbls, ax=ax, 
                        palette=COLOR_MAP, s=100, alpha=0.8, edgecolor="white", linewidth=0.5)
        
        # --- STYLE PROFESSIONNEL DU CADRE ---
        ax.set_title(f"MODÈLE EXPERT : {info['snr']}dB", 
                     fontsize=20, 
                     fontweight='bold', 
                     color="#011b34", 
                     pad=10)
        
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.get_legend().remove() # On enlève les légendes locales pour une légende globale
        
        # Bordures
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#bdc3c7')
            spine.set_linewidth(1.5)

    # --- AJOUT D'UNE LÉGENDE UNIQUE ET CLAIRE ---
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=20, 
               title="Types de Signatures Identifiées", title_fontsize=22,
               frameon=True, facecolor='white', shadow=True, borderpad=1)

    # Ajustement final pour éviter les chevauchements
    plt.subplots_adjust(top=0.88, bottom=0.15, hspace=0.3, wspace=0.2)
    
    # Sauvegarde haute qualité
    plt.savefig("tsne_drone_final_fixed.png", dpi=300, facecolor=fig.get_facecolor())
    print("Infographie t-SNE générée avec succès : 'tsne_drone_final_fixed.png'")
    plt.show()

if __name__ == "__main__":
    run_pro_tsne = run_multi_tsne
    run_pro_tsne()
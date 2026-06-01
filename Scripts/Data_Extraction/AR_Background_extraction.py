import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis, skew
# Imports pour les images 
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
# Utilisation du mode 'Agg' pour éviter d'ouvrir des fenêtres graphiques et économiser de la mémoire
matplotlib.use('Agg') 
# CONFIGURATION DES CHEMINS 
    # Chemin vers le dossier contenant les fichiers CSV bruts (raw data)
input_folder = r"./PPP/data_raw"
    # Chemin pour sauvegarder le fichier CSV final contenant les caractéristiques extraites
output_file = r"./PPP/processed data/ML/final_features_dataset.csv"
# Chemin du dossier où seront enregistrées les images des spectrogrammes
spectro_dir = r"./PPP/processed data/DL/spectrograms"


# Taille de chaque segment de signal (10 000 points par échantillon)
segment_size = 10000

def get_labels_from_filename(filename):
    """
    Détermine le Label (type de drone) et le Mode (état de vol) selon le nom du fichier
    """
    drone_id = -1 # Valeur par défaut si non identifié
    mode_id = 0   # Mode par défaut (souvent pour le bruit de fond)

    # Identification du drone selon les premiers caractères du nom de fichier
    if filename.startswith("00000"): 
        drone_id = 0 # 0 correspond au Background (bruit de fond)
    elif filename.startswith("100"): 
        drone_id = 1 # 1 correspond au Bebop
    elif filename.startswith("101"): 
        drone_id = 2 # 2 correspond à l'AR Drone
    elif filename.startswith("110"): 
        drone_id = 3 # 3 correspond au Phantom

    # Si ce n'est pas du bruit de fond, on extrait le mode de vol (bits 3 et 4 du nom)
    if drone_id != 0: 
        mode_bits = filename[3:5] 
        if mode_bits == "00": mode_id = 1
        elif mode_bits == "01": mode_id = 2
        elif mode_bits == "10": mode_id = 3
        elif mode_bits == "11": mode_id = 4

    return drone_id, mode_id

# Liste pour stocker toutes les caractéristiques calculées avant la sauvegarde
all_features = []

print("Démarrage du traitement global (Mode Analysis + Spectrograms Included)...")

# Boucle parcourant chaque fichier dans le dossier source
for filename in os.listdir(input_folder):
    # On ne traite que les fichiers CSV
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        
        # Récupération du label et du mode via la fonction définie plus haut
        label, mode = get_labels_from_filename(filename)
        
        # Si le fichier ne correspond à aucun drone connu, on l'ignore
        if label == -1: continue 

        # Création des dossiers selon le Label 
        # Dictionnaire pour mapper l'ID numérique au nom textuel du dossier
        folder_names = {
            0: "Background activities",
            1: "Bebop drone",
            2: "AR Drone",
            3: "Phantom drone"
        }
        drone_folder_name = folder_names.get(label, "Unknown")
        # Chemin complet du sous-dossier pour le drone actuel
        current_spectro_path = os.path.join(spectro_dir, drone_folder_name)
        # Création du dossier s'il n'existe pas encore
        os.makedirs(current_spectro_path, exist_ok=True)
        
        # Extraction du nom du fichier sans l'extension pour nommer les images
        file_id = filename.split('.')[0] 
        
        
        print(f"Traitement de {filename} | Label: {label} | Mode: {mode}")
        
        # 1. Lecture du signal (on lit la première ligne du CSV et on l'aplatit en tableau)
        df = pd.read_csv(file_path, header=None, nrows=1)
        signal_raw = df.values.flatten()
        
        # 2. Nettoyage & Normalisation
        # Centrage du signal (soustraction de la moyenne)
        signal_raw = signal_raw - np.mean(signal_raw)
        # Normalisation entre -1 et 1 en divisant par la valeur absolue maximale
        max_val = np.max(np.abs(signal_raw))
        if max_val > 0: signal_raw = signal_raw / max_val
        
        # 3. Segmentation du signal en morceaux de 10 000 points
        num_segments = len(signal_raw) // segment_size
        for i in range(num_segments):
            # Extraction du segment actuel
            seg = signal_raw[i*segment_size : (i+1)*segment_size]
            
            # Calcul des caractéristiques statistiques du segment
            m = np.mean(seg)      # Moyenne
            v = np.var(seg)       # Variance
            k = kurtosis(seg)     # Kurtosis (forme de la distribution)
            s = skew(seg)         # Skewness (asymétrie)
            
            # Calcul du PAPR (Peak-to-Average Power Ratio)
            sq_seg = np.square(seg)
            avg_pwr = np.mean(sq_seg)
            # Conversion en décibels (dB)
            papr = 10 * np.log10(np.max(sq_seg) / avg_pwr) if avg_pwr != 0 else 0
            
            # Ajout des statistiques, du label et du mode dans la liste globale
            all_features.append([m, v, k, s, papr, label, mode])

            #  Génération du Spectrogramme pour CHAQUE segment 
            # Création d'une petite figure (1x1 pouce) avec 64 DPI pour une image de 64x64 pixels
            plt.figure(figsize=(1, 1), dpi=64) 
            # Calcul du spectrogramme (Fréquence d'échantillonnage fixée à 40MHz)
            f, t_spec, Sxx = signal.spectrogram(seg, fs=40000000)
            # Affichage en couleur (échelle logarithmique dB) avec la colormap 'viridis'
            plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            # Suppression des axes (X et Y) pour ne garder que l'image pure du signal
            plt.axis('off')
            
            # Construction du nom de l'image (Drone_Mode_Fichier_Segment.png)
            img_name = f"{drone_folder_name}M{mode}{file_id}_Seg{i}.png"
            # Sauvegarde de l'image sans bordures blanches
            plt.savefig(os.path.join(current_spectro_path, img_name), bbox_inches='tight', pad_inches=0)
            # Fermeture de la figure pour libérer la mémoire vive (RAM)
            plt.close() 
            

#  SAUVEGARDE FINALE DES DONNÉES TABULAIRES 
columns = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Label', 'Mode']
# Création d'un DataFrame Pandas avec tous les résultats
df_final = pd.DataFrame(all_features, columns=columns)

# Écriture du fichier CSV final sur le disque
df_final.to_csv(output_file, index=False)

# Affichage des messages de fin
print(f"\nTERMINE ! Dataset créé avec {len(df_final)} exemples.")
print(f"Images enregistrées dans : {spectro_dir}")
print(f"Fichier CSV sauvegardé dans : {output_file}")
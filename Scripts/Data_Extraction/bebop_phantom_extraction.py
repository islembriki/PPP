import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis, skew
# Imports pour la génération d'images ---
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
# Utilisation du mode 'Agg' pour générer les images en arrière-plan sans ouvrir de fenêtres
matplotlib.use('Agg') 


# CONFIGURATION DES CHEMINS RELATIFS
base_path = "./PPP"
# Chemin pour le fichier CSV final dans le dossier data_processed
output_file = os.path.join(base_path, "processed data", "ML", "drone_master_dataset.csv")
# Chemin pour le dossier qui contiendra les images (spectrogrammes)
spectro_dir = os.path.join(base_path, "processed data", "DL", "spectrograms")

# Taille de chaque segment de signal (10 000 points)
segment_size = 10000

# Dictionnaire de correspondance pour les IDs des drones (Niveau 2)
# 1 = Bebop, 3 = Phantom
DRONE_LABELS = {
    "Bebop drone": 1,
    "Phantom drone": 3
}

# Liste pour stocker toutes les caractéristiques extraites
all_features = []

print(">>> Démarrage du Pipeline Global (Accordance + Mode Analysis + Spectrograms) <<<")

# BOUCLE SUR LES TYPES DE DRONES 
for drone_folder, drone_id in DRONE_LABELS.items():
    # Construction du chemin vers le dossier du drone 
    drone_path = os.path.join(base_path, drone_folder)
    
    #  Création d'un sous-dossier par type de drone pour les images 
    drone_name = "Bebop" if drone_id == 1 else "Phantom"
    current_spectro_path = os.path.join(spectro_dir, drone_name)
    # Crée le dossier s'il n'existe pas
    os.makedirs(current_spectro_path, exist_ok=True)
   

    # Si le dossier du drone n'existe pas, on passe au suivant
    if not os.path.exists(drone_path): 
        print(f"Skipping: {drone_folder} (Path not found)")
        continue

    #  BOUCLE SUR LES SOUS-DOSSIERS D'ENREGISTREMENT (Modes de vol) 
    for subfolder in os.listdir(drone_path):
        subfolder_path = os.path.join(drone_path, subfolder)
        # On ne traite que les répertoires
        if not os.path.isdir(subfolder_path): continue

        # DÉTECTION DU MODE DE VOL 
        # Logique basée sur les bits du nom de dossier : 00=Mode1, 01=Mode2, 10=Mode3, 11=Mode4
        mode_id = 0
        if "00" in subfolder: mode_id = 1    # Connecté
        elif "01" in subfolder: mode_id = 2 # Vol stationnaire (Hovering)
        elif "10" in subfolder: mode_id = 3 # En vol (Sans Vidéo)
        elif "11" in subfolder: mode_id = 4 # En vol (Avec Vidéo)

        print(f"Processing: {drone_folder} | Mode {mode_id} | Subfolder: {subfolder}")

        #  BOUCLE SUR LES FICHIERS CSV DANS CHAQUE MODE 
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(subfolder_path, filename)
                print(f"   --> Processing File: {filename} (Drone: {drone_name}, Mode: {mode_id})")
                
                # Sauvegarde du nom du fichier sans l'extension pour nommer les images
                file_id = filename.split('.')[0]
                
                try:
                    # LECTURE : On lit la première ligne du CSV et on l'aplatit en tableau numpy
                    df = pd.read_csv(file_path, header=None, nrows=1)
                    signal_data = df.values.flatten()
                    
                    # NETTOYAGE : On centre le signal en retirant la moyenne
                    signal_data = signal_data - np.mean(signal_data)
                    # NORMALISATION : On divise par la valeur absolue max pour être entre -1 et 1
                    max_val = np.max(np.abs(signal_data))
                    if max_val > 0: signal_data = signal_data / max_val
                    
                    # SEGMENTATION : On découpe le signal en morceaux de 10 000 points
                    num_segments = len(signal_data) // segment_size
                    for i in range(num_segments):
                        # Extraction du segment i
                        seg = signal_data[i*segment_size : (i+1)*segment_size]
                        
                        # EXTRACTION DES 5 CARACTÉRISTIQUES STATISTIQUES
                        m = np.mean(seg)      # Moyenne
                        v = np.var(seg)       # Variance
                        k = kurtosis(seg)     # Kurtosis (forme)
                        s = skew(seg)         # Skewness (asymétrie)
                        
                        # Calcul du PAPR (Peak-to-Average Power Ratio)
                        sq_seg = np.square(seg)
                        avg_pwr = np.mean(sq_seg)
                        # Formule du PAPR en décibels
                        papr = 10 * np.log10(np.max(sq_seg) / avg_pwr) if avg_pwr != 0 else 0
                        
                        # AJOUT À LA LISTE : [Stats] + [ID Drone] + [ID Mode]
                        all_features.append([m, v, k, s, papr, drone_id, mode_id])

                        # Génération et sauvegarde du Spectrogramme pour CHAQUE segment
                        # Création d'une figure de 1x1 pouce à 64 DPI (image de 64x64 pixels)
                        plt.figure(figsize=(1, 1), dpi=64) 
                        # Calcul du spectrogramme (Fréquence d'échantillonnage 40MHz)
                        f, t_spec, Sxx = signal.spectrogram(seg, fs=40000000)
                        # Affichage du spectrogramme en couleurs (dB)
                        plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
                        # On cache les axes (X et Y)
                        plt.axis('off')
                        
                        # Nom unique pour l'image PNG
                        img_name = f"{drone_name}_M{mode_id}_{file_id}_Seg{i}.png"
                        # Sauvegarde physique de l'image
                        plt.savefig(os.path.join(current_spectro_path, img_name), bbox_inches='tight', pad_inches=0)
                        # Fermeture de la figure pour libérer la mémoire vive
                        plt.close() 
                       
                        
                except Exception as e:
                    print(f"Error on {filename}: {e}")

#SAUVEGARDE DU DATASET FINAL
# Définition des colonnes du DataFrame
cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Label', 'Mode']
df_final = pd.DataFrame(all_features, columns=cols)

# Sécurité : Création du dossier de destination s'il n'existe pas
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Sauvegarde du fichier CSV
df_final.to_csv(output_file, index=False)

print(f"\nFINISHED!")
print(f"Final dataset has {len(df_final)} segments.")
print(f"Images generated in: {spectro_dir}")
print(f"File saved to: {output_file}")
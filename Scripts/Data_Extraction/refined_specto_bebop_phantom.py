import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
import time

# Force l'utilisation d'un backend non-interactif pour éviter d'ouvrir des fenêtres et améliorer la vitesse
matplotlib.use('Agg')

# 1. CONFIGURATION DES CHEMINS 
# Liste des dossiers sources contenant les données brutes des drones
INPUT_DRONES = [
    r'./PPP/Bebop drone', 
    r'./PPP/Phantom drone'
]
# Dossier de destination pour les spectrogrammes générés
OUTPUT_DIR = r'./PPP/processed data/DL/spectrograms_refined'

# Paramètres du signal : 10 000 points par segment et fréquence de 40 MHz
SEGMENT_SIZE = 10000
FS = 40000000 

# 2. LA FABRIQUE DE SPECTROGRAMMES

def create_spectrogram(seg, save_path):
    """ Calcule et sauvegarde une image contrastée du signal """
    # Calcul du spectrogramme (TFCT) avec fenêtre de 512 et recouvrement de 256
    f, t, Sxx = signal.spectrogram(seg, fs=FS, nperseg=512, noverlap=256)
    # Conversion de l'échelle de puissance en décibels (dB)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    # Définition des bornes de couleurs pour isoler le signal utile du bruit de fond
    VMIN, VMAX = -110, -40 
    
    # Création d'une figure carrée de 128x128 pixels (2x2 pouces à 64 DPI)
    plt.figure(figsize=(2, 2), dpi=64) 
    # Dessin des données avec la palette 'magma' et lissage 'gouraud'
    plt.pcolormesh(t, f, Sxx_db, vmin=VMIN, vmax=VMAX, shading='gouraud', cmap='magma')
    # Suppression des axes (chiffres X et Y) pour l'entraînement de l'IA
    plt.axis('off')
    # Sauvegarde de l'image finale sans bordures blanches
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # Fermeture de la figure pour libérer la mémoire RAM
    plt.close()

def get_drone_info(filename):
    """ Extrait le nom du drone et le mode de vol à partir du nom du fichier """
    drone_name = "Unknown"
    mode = 1
    # Décodage du type de drone selon le préfixe
    if filename.startswith("100"): drone_name = "Bebop"
    elif filename.startswith("110"): drone_name = "Phantom"
    
    try:
        # Décodage du mode de vol (bits 3 et 4 du nom de fichier)
        mode_bits = filename[3:5]
        if mode_bits == "00": mode = 1
        elif mode_bits == "01": mode = 2
        elif mode_bits == "10": mode = 3
        elif mode_bits == "11": mode = 4
    except: mode = 1 # Mode par défaut en cas d'erreur de lecture
    return drone_name, mode

# 3. BOUCLE D'ÉXÉCUTION PRINCIPALE

# Création du dossier de sortie global s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*50)
print(" SPECTROGRAM FACTORY STARTING")
print("="*50)

# Enregistrement du temps de début pour le calcul de la durée totale
start_time_total = time.time()

# Boucle sur les dossiers de drones (Bebop et Phantom)
for drone_folder in INPUT_DRONES:
    # Vérification si le dossier source existe
    if not os.path.exists(drone_folder):
        print(f" MISSING MAIN FOLDER: {drone_folder}")
        continue

    print(f"\n ENTERING MAIN CATEGORY: {os.path.basename(drone_folder)}")
    
    # Identification des sous-dossiers (les différents enregistrements/modes)
    subs = [d for d in os.listdir(drone_folder) if os.path.isdir(os.path.join(drone_folder, d))]
    print(f" Found {len(subs)} sub-folders to process.")

    # Parcours de chaque sous-dossier
    for sub_idx, sub in enumerate(subs):
        sub_path = os.path.join(drone_folder, sub)
        # Liste de tous les fichiers CSV à l'intérieur
        csv_files = [f for f in os.listdir(sub_path) if f.endswith('.csv')]
        
        print(f"\n   --- Sub-folder [{sub_idx+1}/{len(subs)}]: {sub} ---")
        print(f"    Found {len(csv_files)} CSV files in this folder.")
        
        # Parcours de chaque fichier CSV
        for file_idx, fname in enumerate(csv_files):
            # Récupération des informations du drone
            drone_cat, mode = get_drone_info(fname)
            # Création du sous-dossier de destination 
            save_path_cat = os.path.join(OUTPUT_DIR, drone_cat)
            os.makedirs(save_path_cat, exist_ok=True)

            full_file_path = os.path.join(sub_path, fname)
            print(f"   [File {file_idx+1}/{len(csv_files)}] Processing: {fname}...")

            try:
                # Chargement complet du fichier CSV (signal brut)
                df = pd.read_csv(full_file_path, header=None)
                signal_raw = df.values.flatten()
                
                # Prétraitement : centrage du signal (moyenne à zéro)
                signal_raw = signal_raw - np.mean(signal_raw)
                # Normalisation : mise à l'échelle entre -1 et 1
                max_v = np.max(np.abs(signal_raw))
                if max_v > 0: signal_raw = signal_raw / max_v

                # Calcul du nombre de segments possibles
                num_segments = len(signal_raw) // SEGMENT_SIZE
                
                # Boucle de génération d'images pour chaque segment
                for i in range(num_segments):
                    # Construction du nom de l'image (Drone_Mode_Fichier_Segment.png)
                    img_name = f"{drone_cat}_M{mode}_{fname[:-4]}_Seg{i}.png"
                    final_path = os.path.join(save_path_cat, img_name)
                    
                    # On ne génère l'image que si elle n'existe pas déjà 
                    if not os.path.exists(final_path):
                        # Extraction du segment temporel
                        seg = signal_raw[i*SEGMENT_SIZE : (i+1)*SEGMENT_SIZE]
                        # Appel de la fonction de création d'image
                        create_spectrogram(seg, final_path)
                    
                    # Affichage d'un indicateur de progression tous les 100 segments
                    if (i + 1) % 100 == 0:
                        print(f"       Progress: {i + 1}/{num_segments} images created...")

                print(f"    Finished {fname} ({num_segments} images).")
            
            except Exception as e:
                # Affichage des erreurs éventuelles pour ne pas bloquer tout le script
                print(f"    ERROR on {fname}: {e}")

# Calcul et affichage du temps total d'exécution
end_time_total = time.time()
duration = (end_time_total - start_time_total) / 60

print("\n" + "="*50)
print(f" ALL DONE!")
print(f" Total Time: {duration:.2f} minutes")
print(f" Output: {OUTPUT_DIR}")
print("="*50)
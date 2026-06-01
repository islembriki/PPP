import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

# Utilisation du mode 'Agg' pour générer les images en arrière-plan sans ouvrir de fenêtre (gain de performance et mémoire)
matplotlib.use('Agg') 

# 1. CONFIGURATION DES CHEMINS 
BASE_PATH = "./PPP"
# Chemin vers le dossier contenant les fichiers CSV bruts
INPUT_FOLDER = os.path.join(BASE_PATH, "data_raw")
# Chemin de base pour les fichiers de sortie
OUTPUT_BASE = os.path.join(BASE_PATH, "processed data")
# Chemin spécifique pour stocker les images des spectrogrammes
SPECTRO_DIR = os.path.join(OUTPUT_BASE, "DL", "spectrograms")

# Paramètres de segmentation et fréquence d'échantillonnage (40 MHz)
SEGMENT_SIZE = 10000
FS = 40000000 

# 2. FONCTION DE GÉNÉRATION DU SPECTROGRAMME

def save_refined_spectrogram(seg, save_path):
    """ Génère un spectrogramme contrasté sans axes ni bordures """
    # Calcul du spectrogramme avec des paramètres de précision (nperseg=taille fenêtre, noverlap=recouvrement)
    f, t, Sxx = signal.spectrogram(seg, fs=FS, nperseg=512, noverlap=256)
    
    # Conversion de la puissance du signal en décibels (dB)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    # Définition des seuils Min/Max pour augmenter le contraste et isoler le signal du bruit
    VMIN, VMAX = -110, -40 
    
    # Création d'une figure carrée de 128x128 pixels (2x2 pouces à 64 DPI)
    plt.figure(figsize=(2, 2), dpi=64) 
    # Dessin du spectrogramme avec la palette de couleurs 'magma'
    plt.pcolormesh(t, f, Sxx_db, vmin=VMIN, vmax=VMAX, shading='gouraud', cmap='magma')
    # Suppression des axes (graduations X et Y)
    plt.axis('off')
    
    # Sauvegarde de l'image en supprimant les marges blanches
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # Fermeture de la figure pour libérer la mémoire vive (RAM)
    plt.close()


# 3. IDENTIFICATION DES DRONES SELON LE NOM

def get_info_from_filename(filename):
    """ Analyse le nom du fichier pour extraire l'ID du drone, le mode et le nom du dossier """
    drone_id = -1
    mode_id = 0
    folder_name = "Unknown"

    # Logique d'identification selon les préfixes définis
    if filename.startswith("00000"): 
        drone_id, folder_name = 0, "Background"
    elif filename.startswith("100"): 
        drone_id, folder_name = 1, "Bebop"
    elif filename.startswith("101"): 
        drone_id, folder_name = 2, "AR_Drone"
    elif filename.startswith("110"): 
        drone_id, folder_name = 3, "Phantom"

    # Si c'est un drone (id > 0), on extrait le mode de vol (bits 3 et 4)
    if drone_id > 0: 
        mode_bits = filename[3:5] 
        if mode_bits == "00": mode_id = 1
        elif mode_bits == "01": mode_id = 2
        elif mode_bits == "10": mode_id = 3
        elif mode_bits == "11": mode_id = 4

    return drone_id, mode_id, folder_name


# 4. BOUCLE DE TRAITEMENT

# Création du dossier principal des spectrogrammes s'il n'existe pas
os.makedirs(SPECTRO_DIR, exist_ok=True)

# Vérification de l'existence du dossier contenant les CSV
if not os.path.exists(INPUT_FOLDER):
    print(f" Erreur : Le dossier source n'existe pas : {INPUT_FOLDER}")
else:
    print(f" Démarrage de l'extraction des spectrogrammes...")
    # Liste de tous les fichiers .csv présents dans le dossier source
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    print(f" {len(files)} fichiers CSV trouvés.")

    # Parcours de chaque fichier
    for filename in files:
        # Extraction des informations (Label, Mode, Nom)
        label, mode, drone_folder = get_info_from_filename(filename)
        if label == -1: continue # Ignore les fichiers non identifiés

        # Création du sous-dossier spécifique au drone 
        current_out_path = os.path.join(SPECTRO_DIR, drone_folder)
        os.makedirs(current_out_path, exist_ok=True)

        file_path = os.path.join(INPUT_FOLDER, filename)
        file_id = filename.split('.')[0] # Nom du fichier sans extension

        try:
            # Chargement de toutes les colonnes du CSV
            df = pd.read_csv(file_path, header=None)
            # Transformation des données en un seul vecteur (tableau 1D)
            signal_raw = df.values.flatten()
            
            # Centrage du signal (soustraction de la moyenne)
            signal_raw = signal_raw - np.mean(signal_raw)
            # Normalisation entre -1 et 1
            max_val = np.max(np.abs(signal_raw))
            if max_val > 0: signal_raw = signal_raw / max_val
            
            # Calcul du nombre de segments de 10 000 points disponibles
            num_segments = len(signal_raw) // SEGMENT_SIZE
            print(f" Traitement : {filename} ({num_segments} segments)")

            # Boucle sur chaque segment pour générer l'image correspondante
            for i in range(num_segments):
                seg = signal_raw[i*SEGMENT_SIZE : (i+1)*SEGMENT_SIZE]
                
                # Construction du nom de l'image finale
                img_name = f"{drone_folder}_M{mode}_{file_id}_Seg{i}.png"
                save_path = os.path.join(current_out_path, img_name)
                
                # Génère et sauvegarde l'image uniquement si elle n'existe pas déjà (gain de temps)
                if not os.path.exists(save_path):
                    save_refined_spectrogram(seg, save_path)

        except Exception as e:
            # Affiche l'erreur si un fichier pose problème 
            print(f" Erreur sur {filename}: {e}")

    print(f"\n TERMINÉ ! Images sauvegardées dans : {SPECTRO_DIR}")
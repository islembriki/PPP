import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

# Empêche les fenêtres popup et améliore les performances
matplotlib.use('Agg') 

# ==========================================
# 1. CONFIGURATION DES CHEMINS (CORRIGÉS)
# ==========================================
# J'ai utilisé le chemin détecté dans votre capture d'écran (garba)
BASE_PATH = r"C:\Users\garba\Desktop\RT3 2025-2026\semestre2\ppp\Projet_Drone"
INPUT_FOLDER = os.path.join(BASE_PATH, "data_raw")
OUTPUT_BASE = os.path.join(BASE_PATH, "data_processed")
SPECTRO_DIR = os.path.join(OUTPUT_BASE, "spectrograms")

SEGMENT_SIZE = 10000
FS = 40000000 # 40MHz

# ==========================================
# 2. FONCTION DE GÉNÉRATION DU SPECTROGRAMME
# ==========================================
def save_refined_spectrogram(seg, save_path):
    """ Génère un spectrogramme contrasté sans axes ni bordures """
    f, t, Sxx = signal.spectrogram(seg, fs=FS, nperseg=512, noverlap=256)
    
    # Conversion en dB
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    # VMIN/VMAX pour isoler le signal du bruit de fond
    VMIN, VMAX = -110, -40 
    
    plt.figure(figsize=(2, 2), dpi=64) # Image 128x128
    plt.pcolormesh(t, f, Sxx_db, vmin=VMIN, vmax=VMAX, shading='gouraud', cmap='magma')
    plt.axis('off')
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ==========================================
# 3. IDENTIFICATION DES DRONES SELON LE NOM
# ==========================================
def get_info_from_filename(filename):
    drone_id = -1
    mode_id = 0
    folder_name = "Unknown"

    if filename.startswith("00000"): 
        drone_id, folder_name = 0, "Background"
    elif filename.startswith("100"): 
        drone_id, folder_name = 1, "Bebop"
    elif filename.startswith("101"): 
        drone_id, folder_name = 2, "AR_Drone"
    elif filename.startswith("110"): 
        drone_id, folder_name = 3, "Phantom"

    if drone_id > 0: 
        mode_bits = filename[3:5] 
        if mode_bits == "00": mode_id = 1
        elif mode_bits == "01": mode_id = 2
        elif mode_bits == "10": mode_id = 3
        elif mode_bits == "11": mode_id = 4

    return drone_id, mode_id, folder_name

# ==========================================
# 4. BOUCLE DE TRAITEMENT
# ==========================================
os.makedirs(SPECTRO_DIR, exist_ok=True)

# Vérification si le dossier d'entrée existe
if not os.path.exists(INPUT_FOLDER):
    print(f"❌ Erreur : Le dossier source n'existe pas : {INPUT_FOLDER}")
else:
    print(f"🚀 Démarrage de l'extraction des spectrogrammes...")
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    print(f"📂 {len(files)} fichiers CSV trouvés.")

    for filename in files:
        label, mode, drone_folder = get_info_from_filename(filename)
        if label == -1: continue

        # Dossier de sortie spécifique au drone (ex: data_processed/spectrograms/Bebop)
        current_out_path = os.path.join(SPECTRO_DIR, drone_folder)
        os.makedirs(current_out_path, exist_ok=True)

        file_path = os.path.join(INPUT_FOLDER, filename)
        file_id = filename.split('.')[0]

        try:
            # Chargement des données
            df = pd.read_csv(file_path, header=None)
            signal_raw = df.values.flatten()
            
            # Normalisation
            signal_raw = signal_raw - np.mean(signal_raw)
            max_val = np.max(np.abs(signal_raw))
            if max_val > 0: signal_raw = signal_raw / max_val
            
            num_segments = len(signal_raw) // SEGMENT_SIZE
            print(f"📦 Traitement : {filename} ({num_segments} segments)")

            for i in range(num_segments):
                seg = signal_raw[i*SEGMENT_SIZE : (i+1)*SEGMENT_SIZE]
                
                img_name = f"{drone_folder}_M{mode}_{file_id}_Seg{i}.png"
                save_path = os.path.join(current_out_path, img_name)
                
                # Génère l'image si elle n'existe pas déjà
                if not os.path.exists(save_path):
                    save_refined_spectrogram(seg, save_path)

        except Exception as e:
            print(f"❌ Erreur sur {filename}: {e}")

    print(f"\n✅ TERMINÉ ! Images sauvegardées dans : {SPECTRO_DIR}")
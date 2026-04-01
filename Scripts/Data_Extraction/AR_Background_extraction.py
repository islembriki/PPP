import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis, skew
# --- AJOUT 1 : Imports pour les images ---
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
matplotlib.use('Agg') # Empêche les fuites de mémoire
# -----------------------------------------

# --- CONFIGURATION ---
input_folder = r"C:\Users\garba\Desktop\RT3 2025-2026\semestre2\ppp\Projet_Drone\data_raw"
output_file = r"C:\Users\garba\Desktop\RT3 2025-2026\semestre2\ppp\Projet_Drone\data_processed\final_features_dataset.csv"
# --- AJOUT 2 : Dossier pour les spectrogrammes ---
spectro_dir = r"C:\Users\garba\Desktop\RT3 2025-2026\semestre2\ppp\Projet_Drone\data_processed\spectrograms"
# -------------------------------------------------
segment_size = 10000

def get_labels_from_filename(filename):
    """
    Détermine le Label et le Mode selon le Figure 7
    """
    drone_id = -1
    mode_id = 0 

    if filename.startswith("00000"): 
        drone_id = 0
    elif filename.startswith("100"): 
        drone_id = 1
    elif filename.startswith("101"): 
        drone_id = 2
    elif filename.startswith("110"): 
        drone_id = 3

    if drone_id != 0: 
        mode_bits = filename[3:5] 
        if mode_bits == "00": mode_id = 1
        elif mode_bits == "01": mode_id = 2
        elif mode_bits == "10": mode_id = 3
        elif mode_bits == "11": mode_id = 4

    return drone_id, mode_id

all_features = []

print("Démarrage du traitement global (Mode Analysis + Spectrograms Included)...")

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_folder, filename)
        
        label, mode = get_labels_from_filename(filename)
        
        if label == -1: continue 

        # --- AJOUT 3 : Création des dossiers selon le Label (Aziza's naming) ---
        # On définit le nom du dossier selon le drone détecté
        folder_names = {
            0: "Background activities",
            1: "Bebop drone",
            2: "AR Drone",
            3: "Phantom drone"
        }
        drone_folder_name = folder_names.get(label, "Unknown")
        current_spectro_path = os.path.join(spectro_dir, drone_folder_name)
        os.makedirs(current_spectro_path, exist_ok=True)
        
        file_id = filename.split('.')[0] # On garde le nom du fichier pour l'image
        # ----------------------------------------------------------------------
        
        print(f"Traitement de {filename} | Label: {label} | Mode: {mode}")
        
        # 1. Lecture
        df = pd.read_csv(file_path, header=None, nrows=1)
        signal_raw = df.values.flatten()
        
        # 2. Nettoyage & Normalisation
        signal_raw = signal_raw - np.mean(signal_raw)
        max_val = np.max(np.abs(signal_raw))
        if max_val > 0: signal_raw = signal_raw / max_val
        
        # 3. Segmentation & Extraction
        num_segments = len(signal_raw) // segment_size
        for i in range(num_segments):
            seg = signal_raw[i*segment_size : (i+1)*segment_size]
            
            # Caractéristiques (Stats)
            m = np.mean(seg)
            v = np.var(seg)
            k = kurtosis(seg)
            s = skew(seg)
            
            # PAPR
            sq_seg = np.square(seg)
            avg_pwr = np.mean(sq_seg)
            papr = 10 * np.log10(np.max(sq_seg) / avg_pwr) if avg_pwr != 0 else 0
            
            # Ajout des données avec Label ET Mode
            all_features.append([m, v, k, s, papr, label, mode])

            # --- AJOUT 4 : Génération du Spectrogramme pour CHAQUE segment ---
            plt.figure(figsize=(1, 1), dpi=64) 
            f, t_spec, Sxx = signal.spectrogram(seg, fs=40000000)
            plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            plt.axis('off')
            
            # Nom de l'image : ex AR Drone_M2_10101H_Seg5.png
            img_name = f"{drone_folder_name}M{mode}{file_id}_Seg{i}.png"
            plt.savefig(os.path.join(current_spectro_path, img_name), bbox_inches='tight', pad_inches=0)
            plt.close() 
            # -----------------------------------------------------------------

# --- SAUVEGARDE FINALE ---
columns = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Label', 'Mode']
df_final = pd.DataFrame(all_features, columns=columns)

df_final.to_csv(output_file, index=False)

print(f"\nTERMINE ! Dataset créé avec {len(df_final)} exemples.")
print(f"Images enregistrées dans : {spectro_dir}")
print(f"Fichier CSV sauvegardé dans : {output_file}")
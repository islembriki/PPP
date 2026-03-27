import pandas as pd
import numpy as np
import os
from scipy.stats import kurtosis, skew
# --- ADDITION 1: New imports for images ---
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
matplotlib.use('Agg') # Prevents memory leaks
# ------------------------------------------

# --- 1. CONFIGURATION (Your Folders) ---
base_path = r"C:\Users\HP\Desktop\ppp"
output_file = os.path.join(base_path, "data_processed", "drone_master_dataset.csv")
# --- ADDITION 2: Path for images ---
spectro_dir = os.path.join(base_path, "data_processed", "spectrograms")
# -----------------------------------
segment_size = 10000

# Level 2 Labels (Matches teammate's ID mapping)
# 1 = Bebop, 3 = Phantom
DRONE_LABELS = {
    "Bebop drone": 1,
    "Phantom drone": 3
}

all_features = []

print(">>> Démarrage du Pipeline Global (Accordance + Mode Analysis + Spectrograms) <<<")

# --- 2. LOOP THROUGH DRONE TYPES ---
for drone_folder, drone_id in DRONE_LABELS.items():
    drone_path = os.path.join(base_path, drone_folder)
    
    # --- ADDITION 3: Create folder for each drone type ---
    drone_name = "Bebop" if drone_id == 1 else "Phantom"
    current_spectro_path = os.path.join(spectro_dir, drone_name)
    os.makedirs(current_spectro_path, exist_ok=True)
    # -----------------------------------------------------

    if not os.path.exists(drone_path): 
        print(f"Skipping: {drone_folder} (Path not found)")
        continue

    # --- 3. LOOP THROUGH RECORDING FOLDERS (Modes) ---
    for subfolder in os.listdir(drone_path):
        subfolder_path = os.path.join(drone_path, subfolder)
        if not os.path.isdir(subfolder_path): continue

        # --- LEVEL 3: MODE DETECTION ---
        # Logic from Figure 7: 00=Mode1, 01=Mode2, 10=Mode3, 11=Mode4
        mode_id = 0
        if "00" in subfolder: mode_id = 1    # Connected
        elif "01" in subfolder: mode_id = 2 # Hovering
        elif "10" in subfolder: mode_id = 3 # Flying (No Video)
        elif "11" in subfolder: mode_id = 4 # Flying (With Video)

        print(f"Processing: {drone_folder} | Mode {mode_id} | Subfolder: {subfolder}")

        # --- 4. LOOP THROUGH CSV FILES ---
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(subfolder_path, filename)
                print(f"   --> Processing File: {filename} (Drone: {drone_name}, Mode: {mode_id})")
                # --- ADDITION 4: Save filename for image naming ---
                file_id = filename.split('.')[0]
                # --------------------------------------------------
                try:
                    # LOAD (Matches teammate's nrows=1 / flatten)
                    df = pd.read_csv(file_path, header=None, nrows=1)
                    signal_data = df.values.flatten()
                    
                    # CLEAN & NORMALIZE (Matches teammate's math)
                    signal_data = signal_data - np.mean(signal_data)
                    max_val = np.max(np.abs(signal_data))
                    if max_val > 0: signal_data = signal_data / max_val
                    
                    # SEGMENT & EXTRACT
                    num_segments = len(signal_data) // segment_size
                    for i in range(num_segments):
                        seg = signal_data[i*segment_size : (i+1)*segment_size]
                        
                        # 5 FEATURES (Matches teammate's math exactly)
                        m = np.mean(seg)
                        v = np.var(seg)
                        k = kurtosis(seg)
                        s = skew(seg)
                        
                        sq_seg = np.square(seg)
                        avg_pwr = np.mean(sq_seg)
                        # PAPR Formula (Matches teammate)
                        papr = 10 * np.log10(np.max(sq_seg) / avg_pwr) if avg_pwr != 0 else 0
                        
                        # ADD TO MASTER LIST
                        # Format: [Features] + [Drone ID] + [Mode ID]
                        all_features.append([m, v, k, s, papr, drone_id, mode_id])

                        # --- ADDITION 5: Create and Save Spectrogram for EVERY segment ---
                        plt.figure(figsize=(1, 1), dpi=64) 
                        f, t_spec, Sxx = signal.spectrogram(seg, fs=40000000)
                        plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
                        plt.axis('off')
                        
                        img_name = f"{drone_name}_M{mode_id}_{file_id}_Seg{i}.png"
                        plt.savefig(os.path.join(current_spectro_path, img_name), bbox_inches='tight', pad_inches=0)
                        plt.close() 
                        # -----------------------------------------------------------------
                        
                except Exception as e:
                    print(f"Error on {filename}: {e}")

# --- 5. SAVE FINAL DATASET ---
# Columns: First 6 match teammate's names and order exactly.
cols = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Label', 'Mode']
df_final = pd.DataFrame(all_features, columns=cols)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df_final.to_csv(output_file, index=False)

print(f"\nFINISHED!")
print(f"Final dataset has {len(df_final)} segments.")
print(f"Images generated in: {spectro_dir}")
print(f"File saved to: {output_file}")
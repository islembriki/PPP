import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
import time

# Force non-interactive backend
matplotlib.use('Agg')

# ==========================================
# 1. PATHS
# ==========================================
INPUT_DRONES = [
    r'C:\Users\HP\Desktop\ppp_perso\ppp\Bebop drone', 
    r'C:\Users\HP\Desktop\ppp_perso\ppp\Phantom drone'
]
OUTPUT_DIR = r'C:\Users\HP\Desktop\ppp_perso\ppp\data_processed\spectrograms_refined'

SEGMENT_SIZE = 10000
FS = 40000000 

# ==========================================
# 2. THE SPECTROGRAM FACTORY
# ==========================================
def create_spectrogram(seg, save_path):
    f, t, Sxx = signal.spectrogram(seg, fs=FS, nperseg=512, noverlap=256)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    VMIN, VMAX = -110, -40 
    
    plt.figure(figsize=(2, 2), dpi=64) 
    plt.pcolormesh(t, f, Sxx_db, vmin=VMIN, vmax=VMAX, shading='gouraud', cmap='magma')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_drone_info(filename):
    drone_name = "Unknown"
    mode = 1
    if filename.startswith("100"): drone_name = "Bebop"
    elif filename.startswith("110"): drone_name = "Phantom"
    
    try:
        mode_bits = filename[3:5]
        if mode_bits == "00": mode = 1
        elif mode_bits == "01": mode = 2
        elif mode_bits == "10": mode = 3
        elif mode_bits == "11": mode = 4
    except: mode = 1
    return drone_name, mode

# ==========================================
# 3. MAIN EXECUTION LOOP (WITH DETAILED DEBUG)
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*50)
print("🚀 SPECTROGRAM FACTORY STARTING")
print("="*50)

start_time_total = time.time()

for drone_folder in INPUT_DRONES:
    if not os.path.exists(drone_folder):
        print(f"⚠️ MISSING MAIN FOLDER: {drone_folder}")
        continue

    print(f"\n📂 ENTERING MAIN CATEGORY: {os.path.basename(drone_folder)}")
    
    # Get subfolders
    subs = [d for d in os.listdir(drone_folder) if os.path.isdir(os.path.join(drone_folder, d))]
    print(f"🔎 Found {len(subs)} sub-folders to process.")

    for sub_idx, sub in enumerate(subs):
        sub_path = os.path.join(drone_folder, sub)
        csv_files = [f for f in os.listdir(sub_path) if f.endswith('.csv')]
        
        print(f"\n   --- Sub-folder [{sub_idx+1}/{len(subs)}]: {sub} ---")
        print(f"   📄 Found {len(csv_files)} CSV files in this folder.")
        
        for file_idx, fname in enumerate(csv_files):
            drone_cat, mode = get_drone_info(fname)
            save_path_cat = os.path.join(OUTPUT_DIR, drone_cat)
            os.makedirs(save_path_cat, exist_ok=True)

            full_file_path = os.path.join(sub_path, fname)
            print(f"   [File {file_idx+1}/{len(csv_files)}] Processing: {fname}...")

            try:
                # Load row
                df = pd.read_csv(full_file_path, header=None)
                signal_raw = df.values.flatten()
                
                # Pre-processing
                signal_raw = signal_raw - np.mean(signal_raw)
                max_v = np.max(np.abs(signal_raw))
                if max_v > 0: signal_raw = signal_raw / max_v

                num_segments = len(signal_raw) // SEGMENT_SIZE
                
                # Progress tracking for segments
                for i in range(num_segments):
                    img_name = f"{drone_cat}_M{mode}_{fname[:-4]}_Seg{i}.png"
                    final_path = os.path.join(save_path_cat, img_name)
                    
                    if not os.path.exists(final_path):
                        seg = signal_raw[i*SEGMENT_SIZE : (i+1)*SEGMENT_SIZE]
                        create_spectrogram(seg, final_path)
                    
                    # Print progress every 100 segments so terminal stays active
                    if (i + 1) % 100 == 0:
                        print(f"      ⏳ Progress: {i + 1}/{num_segments} images created...")

                print(f"   ✅ Finished {fname} ({num_segments} images).")
            
            except Exception as e:
                print(f"   ❌ ERROR on {fname}: {e}")

end_time_total = time.time()
duration = (end_time_total - start_time_total) / 60

print("\n" + "="*50)
print(f"🎉 ALL DONE!")
print(f"⏱️ Total Time: {duration:.2f} minutes")
print(f"📂 Output: {OUTPUT_DIR}")
print("="*50)
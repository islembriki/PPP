import pandas as pd
import numpy as np
import os
import re
from scipy.stats import kurtosis, skew
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================
# CHANGE THESE TWO LINES FOR EACH FOLDER RUN
# ============================================
single_folder_path = r"C:\Users\HP\Desktop\ppp_perso\ppp\Bebop drone\RF Data_10011_L"
output_file        = r"C:\Users\HP\Desktop\PPP\processed data\ML\bebop_10011L.csv"
# ============================================

segment_size = 10000
all_features = []

log("="*60)
log(f"Testing single folder: {single_folder_path}")
log("="*60)

# Mode detection
subfolder = os.path.basename(single_folder_path)
match = re.search(r'(\d{5})', subfolder)
if match:
    binary_part = match.group(1)
    mode_bits   = binary_part[3:5]
    if mode_bits == "00": mode_id = 1
    elif mode_bits == "01": mode_id = 2
    elif mode_bits == "10": mode_id = 3
    elif mode_bits == "11": mode_id = 4
    else: mode_id = 0
else:
    log("✗ Could not detect mode!")
    exit()

log(f"Folder    : {subfolder}")
log(f"binary_part = '{binary_part}' → mode_bits = '{mode_bits}' → Mode {mode_id}")
log("="*60)

csv_files = sorted([f for f in os.listdir(single_folder_path) if f.endswith(".csv")])
log(f"Found {len(csv_files)} CSV files\n")

for file_idx, filename in enumerate(csv_files):
    file_path = os.path.join(single_folder_path, filename)
    log(f"[{file_idx+1}/{len(csv_files)}] {filename} | Label=1 | Mode={mode_id}")

    try:
        df_raw      = pd.read_csv(file_path, header=None, nrows=1)
        signal_data = df_raw.values.flatten()

        signal_data = signal_data - np.mean(signal_data)
        max_val     = np.max(np.abs(signal_data))
        if max_val > 0:
            signal_data = signal_data / max_val

        num_segments = len(signal_data) // segment_size
        log(f"  → {num_segments} segments")

        for i in range(num_segments):
            seg  = signal_data[i*segment_size:(i+1)*segment_size]
            m    = np.mean(seg)
            v    = np.var(seg)
            k    = kurtosis(seg)
            s    = skew(seg)
            sq   = np.square(seg)
            avg  = np.mean(sq)
            papr = 10 * np.log10(np.max(sq) / avg) if avg != 0 else 0
            all_features.append([m, v, k, s, papr, 1, mode_id])

        log(f"  ✓ Total so far: {len(all_features)}")

    except Exception as e:
        log(f"  ✗ ERROR: {e}")

# Save
cols     = ['Mean', 'Variance', 'Kurtosis', 'Skewness', 'PAPR', 'Label', 'Mode']
df_out   = pd.DataFrame(all_features, columns=cols)
df_out.to_csv(output_file, index=False)

log("="*60)
log(f"✓ Done! {len(all_features)} segments saved to {output_file}")
log(f"Mode distribution:")
print(df_out['Mode'].value_counts().sort_index())
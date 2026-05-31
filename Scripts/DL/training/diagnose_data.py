import os
import re
from collections import Counter, defaultdict

FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Background\Background",
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\AR_Drone\AR_Drone",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom\Phantom"
]

BUI_MAP = {
    "00000": 0, "10000": 1, "10001": 2, "10010": 3, "10011": 4,
    "10100": 5, "10101": 6, "10110": 7, "10111": 8,
    "11000": 9, "11001": 10, "11010": 11, "11011": 12
}

CLASS_NAMES = ["Background", "Bebop_1", "Bebop_2", "Bebop_3", "Bebop_4",
               "Phantom_1", "Phantom_2", "Phantom_3", "Phantom_4",
               "AR_Drone_1", "AR_Drone_2", "AR_Drone_3", "AR_Drone_4"]

class_counts = Counter()
snr_class_counts = defaultdict(lambda: Counter())
total_files = 0
bad_files = 0

print("🔍 Scanning data folders...\n")

for folder_path in FOLDERS:
    if not os.path.exists(folder_path):
        print(f"❌ Folder NOT FOUND: {folder_path}")
        continue

    folder_name = os.path.basename(folder_path)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    print(f"📁 {folder_name}: {len(files)} files")

    for f in files:
        total_files += 1
        # Extract BUI code (5-bit binary)
        match = re.search(r'([01]{5})', f)
        if match:
            bui = match.group(1)
            label = BUI_MAP.get(bui, -1)
            if label != -1:
                class_counts[label] += 1
                # Try to extract SNR if it's in the filename
                snr_match = re.search(r'SNR_(-?\d+)', f)
                snr = snr_match.group(1) if snr_match else "unknown"
                snr_class_counts[snr][label] += 1
            else:
                bad_files += 1
                print(f"  ⚠️  Unknown BUI: {f}")
        else:
            bad_files += 1
            print(f"  ⚠️  No BUI code found: {f}")

print(f"\n{'='*60}")
print(f"📊 TOTAL FILES: {total_files} | UNRECOGNIZED: {bad_files}")
print(f"{'='*60}\n")

print("📈 CLASS DISTRIBUTION (overall):")
print("-" * 60)
for class_id in sorted(class_counts.keys()):
    count = class_counts[class_id]
    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"
    print(f"  Class {class_id:2d} ({class_name:20s}): {count:6d} samples")

print(f"\n📊 TOTAL SAMPLES: {sum(class_counts.values())}")
print(f"Imbalance Ratio: {max(class_counts.values()) / (min(class_counts.values()) + 1e-6):.2f}x")

if snr_class_counts:
    print(f"\n📈 DISTRIBUTION BY SNR:")
    print("-" * 60)
    for snr in sorted(snr_class_counts.keys()):
        print(f"\nSNR {snr}dB:")
        for class_id in sorted(snr_class_counts[snr].keys()):
            count = snr_class_counts[snr][class_id]
            print(f"  Class {class_id}: {count}")
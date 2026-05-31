import os
import re
from collections import Counter, defaultdict
import sys

# Add paths module
sys.path.insert(0, os.path.dirname(__file__))
from paths import FOLDERS_LIST, verify_paths

# ════════════════════════════════════════════════════════════════
# DATA DIAGNOSIS SCRIPT
# ════════════════════════════════════════════════════════════════

# BUI mapping (same as in data_manager.py)
BUI_MAP = {
    "00000": 0, "10000": 1, "10001": 2, "10010": 3, "10011": 4,
    "10100": 5, "10101": 6, "10110": 7, "10111": 8,
    "11000": 9, "11001": 10, "11010": 11, "11011": 12
}

CLASS_NAMES = [
    "Background",
    "Bebop M1", "Bebop M2", "Bebop M3", "Bebop M4",
    "AR M1",    "AR M2",    "AR M3",    "AR M4",
    "Phantom M1", "Phantom M2", "Phantom M3", "Phantom M4"
]

def diagnose_data(folders_list):
    """
    Analyzes the dataset structure and prints detailed statistics.

    Args:
        folders_list: List of folder paths containing PNG files
    """

    print("\n" + "=" * 80)
    print("🔍 DATA DIAGNOSIS REPORT")
    print("=" * 80 + "\n")

    # Initialize counters
    class_counts = Counter()
    snr_class_counts = defaultdict(lambda: Counter())
    total_files = 0
    bad_files = 0
    folder_details = {}

    # Scan each folder
    print("📁 Scanning folders...\n")

    for folder_path in folders_list:
        if not os.path.exists(folder_path):
            print(f"❌ Folder NOT FOUND: {folder_path}\n")
            continue

        folder_name = os.path.basename(folder_path)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

        print(f"✅ {folder_name:15s}: {len(files):6d} PNG files")
        folder_details[folder_name] = {
            "path": folder_path,
            "file_count": len(files),
            "classes": Counter()
        }

        # Process each file
        for f in files:
            total_files += 1

            # Extract BUI code (5-bit binary)
            match = re.search(r'([01]{5})', f)
            if match:
                bui = match.group(1)
                label = BUI_MAP.get(bui, -1)

                if label != -1:
                    class_counts[label] += 1
                    folder_details[folder_name]["classes"][label] += 1

                    # Try to extract SNR if present in filename
                    snr_match = re.search(r'SNR_(-?\d+)', f)
                    snr = snr_match.group(1) if snr_match else "unknown"
                    snr_class_counts[snr][label] += 1
                else:
                    bad_files += 1
            else:
                bad_files += 1

    print()

    # ════════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total PNG files found:      {total_files}")
    print(f"Unrecognized files:         {bad_files}")
    print(f"Valid labeled files:        {sum(class_counts.values())}")
    print()

    # ════════════════════════════════════════════════════════════════
    # CLASS DISTRIBUTION (Overall)
    # ════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("📈 CLASS DISTRIBUTION (Overall)")
    print("=" * 80)
    print(f"{'Class ID':>8} | {'Class Name':<20} | {'Count':>10} | {'%':>6}\n")
    print("─" * 80)

    total_samples = sum(class_counts.values())

    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown_{class_id}"
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{class_id:8d} | {class_name:<20} | {count:10d} | {percentage:6.2f}%")

    print("─" * 80)
    print(f"{'TOTAL':<30} | {total_samples:10d} | {100.00:6.2f}%\n")

    # ════════════════════════════════════════════════════════════════
    # CLASS IMBALANCE ANALYSIS
    # ════════════════════════════════════════════════════════════════

    imbalance_ratio = 1.0  # ✅ Initialize to avoid UnboundLocalError
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values()) if class_counts else 1
        imbalance_ratio = max_count / (min_count + 1e-6)

        print("=" * 80)
        print("⚖️  CLASS IMBALANCE ANALYSIS")
        print("=" * 80)
        print(f"Most common class:    {max_count} samples")
        print(f"Least common class:   {min_count} samples")
        print(f"Imbalance ratio:      {imbalance_ratio:.2f}x")
        print()

        if imbalance_ratio > 10:
            print("⚠️  WARNING: High class imbalance detected!")
            print("   → Use WeightedRandomSampler during training")
        elif imbalance_ratio > 3:
            print("⚠️  CAUTION: Moderate class imbalance detected")
            print("   → Consider using class weights")
        else:
            print("✅ Class distribution is reasonably balanced")
        print()

    # ════════════════════════════════════════════════════════════════
    # PER-FOLDER BREAKDOWN
    # ════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("📁 FILES PER FOLDER")
    print("=" * 80)

    for folder_name, details in folder_details.items():
        print(f"\n{folder_name}:")
        print(f"  Path: {details['path']}")
        print(f"  Total PNG files: {details['file_count']}")

        if details['classes']:
            print(f"  Classes present: {sorted(details['classes'].keys())}")
            for class_id in sorted(details['classes'].keys()):
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown_{class_id}"
                count = details['classes'][class_id]
                print(f"    - Class {class_id} ({class_name}): {count} files")
        else:
            print(f"  ⚠️  No recognized files in this folder!")

    print()

    # ════════════════════════════════════════════════════════════════
    # SNR DISTRIBUTION (if present in filenames)
    # ════════════════════════════════════════════════════════════════

    if snr_class_counts:
        print("=" * 80)
        print("📊 DISTRIBUTION BY SNR (if encoded in filenames)")
        print("=" * 80)

        for snr in sorted(snr_class_counts.keys(),
                          key=lambda x: int(x) if x != "unknown" else -999):
            total_snr = sum(snr_class_counts[snr].values())
            print(f"\nSNR {snr}dB: {total_snr} total samples")

            for class_id in sorted(snr_class_counts[snr].keys()):
                count = snr_class_counts[snr][class_id]
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown_{class_id}"
                print(f"  Class {class_id:2d} ({class_name:15s}): {count:6d} samples")

    else:
        print("=" * 80)
        print("ℹ️  SNR NOT ENCODED IN FILENAMES")
        print("=" * 80)
        print("SNR values are not found in filenames.")
        print("They are likely added during training via inject_noise() in data_manager.py\n")

    # ════════════════════════════════════════════════════════════════
    # RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════════

    print("=" * 80)
    print("💡 RECOMMENDATIONS FOR TRAINING")
    print("=" * 80)

    recommendations = []

    if imbalance_ratio > 10:
        recommendations.append(
            "• Use WeightedRandomSampler to handle class imbalance\n"
            "  (Code already implements this ✅)"
        )

    if bad_files > total_files * 0.1:
        recommendations.append(
            f"• {bad_files} files ({bad_files/total_files*100:.1f}%) were not recognized\n"
            "  → Check filenames for BUI codes (5-bit binary pattern)"
        )

    if total_samples < 1000:
        recommendations.append(
            f"• Only {total_samples} samples found\n"
            "  → Consider data augmentation during training"
        )

    if not snr_class_counts or "unknown" in snr_class_counts:
        recommendations.append(
            "• SNR values not in filenames\n"
            "  → They are added synthetically during training ✅"
        )

    if recommendations:
        for rec in recommendations:
            print(rec + "\n")
    else:
        print("✅ Dataset looks good! Ready for training.\n")

    print("=" * 80 + "\n")

    return {
        "total_files": total_files,
        "valid_files": sum(class_counts.values()),
        "bad_files": bad_files,
        "class_distribution": dict(class_counts),
        "imbalance_ratio": imbalance_ratio if class_counts else 0
    }

def main():
    """Main diagnostic workflow"""

    print("\n" + "=" * 80)
    print("🔍 DRONE RF DATASET DIAGNOSTIC TOOL")
    print("=" * 80)

    # Verify paths exist
    verify_paths()

    # Run diagnosis
    stats = diagnose_data(FOLDERS_LIST)

    return stats

if __name__ == "__main__":
    main()
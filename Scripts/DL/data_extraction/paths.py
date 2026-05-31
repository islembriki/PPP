import os

# ════════════════════════════════════════════════════════════════
# CENTRALIZED PATH CONFIGURATION
# ════════════════════════════════════════════════════════════════

# 1. WHERE YOUR DATA WILL BE EXTRACTED
EXTRACTED_DATA_BASE = r"C:\PPP\spectrograms"

# 2. INDIVIDUAL DRONE TYPE FOLDERS (after extraction)
FOLDERS = {
    "Background": os.path.join(EXTRACTED_DATA_BASE, "Background", "Background"),
    "Bebop": os.path.join(EXTRACTED_DATA_BASE, "Bebop", "Bebop"),
    "AR_Drone": os.path.join(EXTRACTED_DATA_BASE, "AR_Drone", "AR_Drone"),
    "Phantom": os.path.join(EXTRACTED_DATA_BASE, "Phantom", "Phantom"),
}

# 3. FOLDERS AS LIST (for data_manager.py)
FOLDERS_LIST = list(FOLDERS.values())

# 4. WHERE ZIPs ARE DOWNLOADED
DOWNLOAD_FOLDER = r"C:\Users\SYB lenovo\Downloads"

# 5. ZIP FILE NAMES
ZIP_FILES = {
    "Bebop": "Bebop.zip",
    "Phantom": "Phantom.zip",
    "AR_Drone": "AR_Drone.zip",
    "Background": "Background.zip"
}

# 6. MODEL SAVE LOCATION (trained .pth files)
# ────────────────────────────────────────────────
# Saved here after training completes
# Example: merged_snr_13classes.pth
MODELS_PATH = r"C:\PPP\models"

# 7. RESULTS/LOGS LOCATION (metrics, plots, reports)
# ────────────────────────────────────────────────
# Saved here during/after training:
# - Accuracy/loss plots
# - Confusion matrices
# - Classification reports
# - Training logs
RESULTS_PATH = r"C:\PPP\results"

# ════════════════════════════════════════════════════════════════
# AUTO-CREATE DIRECTORIES
# ════════════════════════════════════════════════════════════════

def create_directories():
    """Create all necessary directories if they don't exist"""
    print("📁 Creating directories...\n")

    # Create models directory
    os.makedirs(MODELS_PATH, exist_ok=True)
    print(f"✅ Models directory: {MODELS_PATH}")

    # Create results subdirectories
    results_subdirs = ["plots", "logs", "evaluation", "summary"]
    for subdir in results_subdirs:
        path = os.path.join(RESULTS_PATH, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"✅ Results subdirectory: {path}")

    print()

# ════════════════════════════════════════════════════════════════
# VERIFY PATHS EXIST
# ════════════════════════════════════════════════════════════════

def verify_paths():
    """Check if all data folders exist and count PNG files"""
    print("🔍 Verifying data paths...\n")

    for name, path in FOLDERS.items():
        if os.path.exists(path):
            file_count = len([f for f in os.listdir(path) if f.lower().endswith('.png')])
            print(f"✅ {name:15s}: {path}")
            print(f"   └─ PNG files: {file_count}")
        else:
            print(f"❌ {name:15s}: NOT FOUND")
            print(f"   └─ Expected: {path}")

    print()

if __name__ == "__main__":
    create_directories()  # Create folders first
    verify_paths()        # Then verify data


# merged_snr_train_10classes.py
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import sys
from tqdm import tqdm
import os

sys.path.insert(0, r"C:\PPP\Scripts\DL\data_extraction")
from data_extraction.paths import FOLDERS
FOLDERS_LIST = list(FOLDERS.values())

# Import from YOUR files (not teammate's)
from model import DroneCNN
from merged_snr_data_manager import DroneRFDatasetMerged, get_smart_splits_merged
from merged_snr_trainer import MergedSNRTrainer

# ════════════════════════════════════════════════════════════════
# MERGED SNR TRAINING - 10 CLASSES (YOUR VERSION)
# ════════════════════════════════════════════════════════════════

# Automatically picks GPU if you have one, otherwise stays on CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📌 Using device: {DEVICE}")
SNR_VALUES = [-10, -5, 0, 5, 10]

# 10-class BUI mapping (only available classes)
BUI_MAP_10 = {
    "00000": 0,
    "10000": 1, "10001": 2, "10010": 3, "10011": 4,
    "10100": 5, "10101": 6, "10110": 7, "10111": 8,
    "11000": 9,
}

CLASS_NAMES_10 = [
    "Background",
    "Bebop M1", "Bebop M2", "Bebop M3", "Bebop M4",
    "AR M1", "AR M2", "AR M3", "AR M4",
    "Phantom M1"
]

def main_10classes():
    """Train merged SNR model with 10 classes"""

    print("\n" + "=" * 80)
    print("🚀 MERGED SNR TRAINING - 10 CLASSES (EFFICIENT)")
    print("=" * 80)
    print(f"📊 SNR values: {SNR_VALUES}")
    print(f"📊 Classes: {len(CLASS_NAMES_10)}")
    print(f"📁 Data folders: {len(FOLDERS_LIST)}")
    print("=" * 80 + "\n")

    trainer = MergedSNRTrainer(DEVICE, approach="10_classes")

    # Load data for all SNRs
    print("📥 Loading data (all SNRs combined)...\n")

    all_samples = []
    for snr in SNR_VALUES:
        print(f"   Loading SNR {snr}dB...")
        ds = DroneRFDatasetMerged(FOLDERS_LIST, target_snr=snr, bui_map=BUI_MAP_10)
        print(f"   ✅ Loaded {len(ds)} samples")
        all_samples.extend(ds.samples)

    # Create merged dataset
    merged_ds = DroneRFDatasetMerged(FOLDERS_LIST, target_snr=None, bui_map=BUI_MAP_10)
    merged_ds.samples = all_samples

    print(f"\n📊 Total merged samples: {len(merged_ds)}\n")

    if len(merged_ds) == 0:
        print("❌ No data loaded!")
        return

    # Split into train/val
    print("📊 Creating train/val split (stratified)...\n")
    train_ds, val_ds = get_smart_splits_merged(merged_ds, train_ratio=0.8)

    # Create weighted sampler
    train_labels = [train_ds.dataset.samples[i]['label'] for i in train_ds.indices]
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=10)

    print("📊 Class distribution in training set:")
    for class_id in range(10):
        count = class_counts[class_id].item()
        print(f"   Class {class_id:2d} ({CLASS_NAMES_10[class_id]:15s}): {count:8d} samples")

    class_weights = 1.0 / (class_counts.float() + 1e-6)
    sample_weights = class_weights[torch.tensor(train_labels)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=256, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n📊 Train batches: {len(train_loader)}")
    print(f"📊 Val batches: {len(val_loader)}\n")

    # Create and train model
    model = DroneCNN(nb_classes=10).to(DEVICE)
    print(f"✅ Model moved to {DEVICE}")

    trainer.train_merged(
        model,
        train_loader,
        val_loader,
        epochs=20,
        snr_range=SNR_VALUES,
        nb_classes=10
    )

    # Save model
    model_path = r"C:\PPP\models\merged_snr_10classes.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved: {model_path}\n")

    # Save training history
    history_path = r"C:\PPP\results\summary\merged_snr_10classes_history.json"
    trainer.save_training_history(history_path)

    return model

if __name__ == "__main__":
    main_10classes()
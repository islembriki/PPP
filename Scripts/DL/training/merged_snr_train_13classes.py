import torch
import numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import sys

# Force IntelliJ to find the data_extraction folder
sys.path.append(r"C:\PPP\Scripts\DL\data_extraction")
from paths import FOLDERS_LIST
from drone_model import DroneCNN
from merged_snr_data_manager import DroneRFDatasetMerged, get_smart_splits_merged
from merged_snr_trainer import MergedSNRTrainer

DEVICE = torch.device("cuda")
SNR_VALUES = [-10, -5, 0, 5, 10]
BUI_MAP_13 = {"00000": 0, "10000": 1, "10001": 2, "10010": 3, "10011": 4, "10100": 5, "10101": 6, "10110": 7, "10111": 8, "11000": 9, "11001": 10, "11010": 11, "11011": 12}

def main():
    print(f"🚀 STARTING FINAL COMPARISON RUN | DEVICE: {DEVICE}")
    trainer = MergedSNRTrainer(DEVICE, approach="purified_merged_study")

    # 1. LOAD AND CLEAN (With Progress Bar)
    all_samples = []
    print("📥 Loading and Purifying Spectrograms...")
    for snr in tqdm(SNR_VALUES, desc="Total Loading Progress"):
        ds = DroneRFDatasetMerged(FOLDERS_LIST, target_snr=snr, bui_map=BUI_MAP_13)
        all_samples.extend(ds.samples)

    merged_ds = DroneRFDatasetMerged(FOLDERS_LIST, target_snr=None, bui_map=BUI_MAP_13)
    merged_ds.samples = all_samples
    print(f"📊 Total Purified Samples: {len(merged_ds)}")

    train_ds, val_ds = get_smart_splits_merged(merged_ds, train_ratio=0.8)

    # 2. INTENSE SNR-AWARE WEIGHTING (With Progress Bar)
    print("⚖️ Calculating SNR-Aware Weights...")
    train_indices = train_ds.indices
    train_labels = [merged_ds.samples[i]['label'] for i in train_indices]
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=13)
    class_weights = 1.0 / (class_counts.float() + 1e-6)

    weights_np = np.zeros(len(train_indices), dtype=np.float64)
    for idx, original_idx in enumerate(tqdm(train_indices, desc="Weighting Samples")):
        s = merged_ds.samples[original_idx]
        w = class_weights[s['label']].item()
        # EXTREME PRIORITY for noise to beat the teammate's score
        if s['snr'] == -10: w *= 4.0
        elif s['snr'] == -5: w *= 2.0
        weights_np[idx] = w

    sampler = WeightedRandomSampler(torch.from_numpy(weights_np), len(weights_np))

    # 3. DATALOADER - Batch 64 (Better for accuracy than 128)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # 4. START TRAINING
    model = DroneCNN(nb_classes=13).to(DEVICE)
    print("🧠 Model Initialized. Starting Training Loop...")
    trainer.train_merged(model, train_loader, val_loader, epochs=20, nb_classes=13)

    torch.save(model.state_dict(), r"C:\PPP\models\FINAL_SUCCESS_MODEL.pth")

if __name__ == "__main__":
    main()
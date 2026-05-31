import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import json
import os
import gc
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════
# FINAL RESEARCH TRAINER (Matched to Teammate's Architecture)
# ════════════════════════════════════════════════════════════════

class MergedSNRTrainer:
    """
    Advanced Trainer for Merged SNR models.
    Includes:
    - Progress visualization (tqdm)
    - Plateau breaking (ReduceLROnPlateau)
    - RAM optimization (gc.collect)
    """

    def __init__(self, device, approach="purified_merged"):
        self.device = device
        self.approach = approach
        self.training_history = {
            "epoch": [],
            "train_loss": [],
            "val_acc": [],
            "learning_rate": [],
            "class_distribution": []
        }

    def train_merged(self, model, train_loader, val_loader, epochs=20,
                     snr_range=None, nb_classes=13):
        """
        Core training loop with automated optimization.
        """
        # 1. Setup Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Added weight_decay

        # 2. Setup Scheduler (The 'Plateau Breaker')
        # This reduces LR by half if accuracy doesn't improve for 2 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )

        model.to(self.device)
        snr_str = f"{snr_range}" if snr_range else "unknown"

        print(f"\n{'='*80}")
        print(f"🎯 STARTING ELITE TRAINING: {self.approach}")
        print(f"📊 SNRs: {snr_str} | Epochs: {epochs} | Classes: {nb_classes}")
        print(f"🚀 Device: {self.device}")
        print(f"{'='*80}\n")

        for epoch in range(epochs):
            # --- RAM & GPU CLEANUP ---
            gc.collect()
            torch.cuda.empty_cache()

            # ════════════════════════════════════════════════════════
            # TRAINING PHASE
            # ════════════════════════════════════════════════════════
            model.train()
            train_loss = 0.0
            train_count = 0

            # Progress Bar for Training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", smoothing=0.1)

            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Metrics
                current_batch_loss = loss.item()
                train_loss += current_batch_loss * len(labels)
                train_count += len(labels)

                # Update the bar with current loss
                train_pbar.set_postfix(loss=f"{current_batch_loss:.4f}")

            avg_train_loss = train_loss / train_count if train_count > 0 else 0

            # ════════════════════════════════════════════════════════
            # VALIDATION PHASE (Progress Bar Fixed)
            # ════════════════════════════════════════════════════════
            model.eval()
            correct = 0
            total = 0
            all_preds = []

            # Progress Bar for Validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val  ]", leave=False)

            with torch.no_grad():
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / total if total > 0 else 0

            # --- STEP THE SCHEDULER ---
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']

            # Class coverage check
            pred_dist = dict(Counter(all_preds))

            # Store history
            self.training_history["epoch"].append(epoch + 1)
            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_acc"].append(val_acc)
            self.training_history["learning_rate"].append(current_lr)
            self.training_history["class_distribution"].append(pred_dist)

            # Final summary print for the epoch
            print(f"✨ EPOCH {epoch+1:02d} | Loss: {avg_train_loss:.4f} | Acc: {val_acc:.2f}% | LR: {current_lr} | Classes: {len(pred_dist)}/{nb_classes}")

            # ════════════════════════════════════════════════════════
            # SAVE CHECKPOINT (Every 5 Epochs)
            # ════════════════════════════════════════════════════════
            if (epoch + 1) % 5 == 0:
                checkpoint_path = f"C:\\PPP\\models\\research_chk_epoch{epoch+1}.pth"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 Checkpoint saved to {checkpoint_path}")

        print(f"\n✅ TRAINING COMPLETE! FINAL ACCURACY: {val_acc:.2f}%\n")
        return model

    def save_training_history(self, output_path):
        """Save detailed training history as JSON for your report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"📊 Training history saved to: {output_path}")
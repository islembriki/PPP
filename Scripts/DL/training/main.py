import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import DroneCNN
from data_manager import DroneRFDataset, get_smart_splits
from trainer import SNRTrainer

# RECOMMANDATION : Garde le CPU pour la stabilité maximale
DEVICE = torch.device("cpu")

# Liste de tes dossiers nettoyés
FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Background\Background",
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\AR_Drone\AR_Drone",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom\Phantom"
]

def main():
    trainer = SNRTrainer(DEVICE)
    
    for snr in [-10]:
        print(f"\n🚀 --- ENTRAINEMENT EXPERT {snr}dB (13 CLASSES) ---")
        full_ds = DroneRFDataset(FOLDERS, target_snr=snr)
        
        if len(full_ds) == 0: continue
        
        train_ds, val_ds = get_smart_splits(full_ds)

        # --- SAMPLER ÉQUILIBRÉ POUR LES 13 CLASSES ---
        train_labels = [train_ds.dataset.samples[i]['label'] for i in train_ds.indices]
        class_counts = torch.bincount(torch.tensor(train_labels), minlength=13)
        # Gestion des classes vides pour éviter division par zéro
        class_weights = 1. / (class_counts.float() + 1e-6)
        sample_weights = class_weights[torch.tensor(train_labels)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        model = DroneCNN(nb_classes=13)
        trainer.train_expert(model, train_loader, val_loader, snr, epochs=10)

if __name__ == "__main__":
    main()
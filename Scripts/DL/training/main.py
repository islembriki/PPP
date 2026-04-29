import torch_directml
from model import DroneCNN
from data_manager import DroneRFDataset, get_smart_splits
from trainer import SNRTrainer
from torch.utils.data import DataLoader

DEVICE = torch_directml.device()
FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop_spectorgrams\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom_spectograms\Phantom",
    r"C:\Users\USER\Desktop\Projet_Drone\data_processed\data_processed\spectrograms\AR Drone",
    r"C:\Users\USER\Desktop\Projet_Drone\data_processed\data_processed\spectrograms\Background activities"
]

def main():
    trainer = SNRTrainer(DEVICE)
    
    # On entraîne pour différents niveaux de bruit simulés
    for snr in [30, 10, 0]: 
        print(f"\n--- ENTRAINEMENT EXPERT {snr}dB ---")
        full_ds = DroneRFDataset(FOLDERS, target_snr=snr)
        
        if len(full_ds) == 0: 
            print("Erreur: Dossiers vides !")
            continue

        train_ds, val_ds = get_smart_splits(full_ds, train_ratio=0.8)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        model = DroneCNN(nb_classes=4)
        trainer.train_expert(model, train_loader, val_loader, snr)

if __name__ == "__main__":
    main()
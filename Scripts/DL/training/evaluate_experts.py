import torch
import torch_directml
from model import DroneCNN, CLASS_NAMES
from data_manager import DroneRFDataset, get_snr_splits
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

DEVICE = torch_directml.device()
FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop_spectorgrams\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom_spectograms\Phantom",
    r"C:\Users\USER\Desktop\Projet_Drone\data_processed\data_processed\spectrograms\AR Drone",
    r"C:\Users\USER\Desktop\Projet_Drone\data_processed\data_processed\spectrograms\Background activities"
]

def evaluate(snr):
    print(f"\n🔍 Analyse de l'expert {snr}dB...")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    full_ds = DroneRFDataset(FOLDERS, target_snr=snr, transform=transform)
    _, val_ds = get_snr_splits(full_ds, train_ratio=0.8)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = DroneCNN(nb_classes=13).to(DEVICE)
    model.load_state_dict(torch.load(f"expert_model_{snr}dB.pth"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix - Expert {snr}dB")
    plt.show()

if __name__ == "__main__":
    # Teste l'expert 30dB par défaut
    evaluate(30)
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

class SNRTrainer:
    def __init__(self, device):
        self.device = device

    def train_expert(self, model, train_loader, val_loader, snr_label, epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.to(self.device)

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Évaluation
            model.eval()
            correct, total, all_preds = 0, 0, []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            print(f"SNR {snr_label}dB | EPOQUE {epoch+1} | Acc: {acc:.2f}% | Prédic: {dict(Counter(all_preds))}")
            torch.save(model.state_dict(), f"expert_{snr_label}dB_13classes.pth")
        return model
import torch
import torch.nn as nn
import torch.optim as optim

class SNRTrainer:
    def __init__(self, device):
        self.device = device

    def train_expert(self, model, train_loader, val_loader, snr_label, epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001) # On garde un LR stable
        
        model.to(self.device)
        best_acc = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if batch_idx % 1000 == 0:
                    print(f"   Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    _, predicted = torch.max(model(images), 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            print(f"📊 TYPE EXPERT | ÉPOQUE {epoch+1} | Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"type_expert_{snr_label}dB.pth")
        return model
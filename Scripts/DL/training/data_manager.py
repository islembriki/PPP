import os, re, torch, random
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

class DroneRFDataset(Dataset):
    def __init__(self, folders, target_snr=None):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)), # On garde 128 pour la précision
            transforms.ToTensor(),
        ])
        self.target_snr = target_snr
        
        # LOGIQUE : 4 TYPES SEULEMENT
        # 0: Background, 1: Bebop, 2: AR Drone, 3: Phantom
        print(f"🔍 Scan des dossiers (Mode 4 TYPES)...")
        for folder in folders:
            if not os.path.exists(folder): continue
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith('.png'):
                        match = re.search(r'([01]{5})', f)
                        if match:
                            bui = match.group(1)
                            # Nouveau mapping simplifié
                            if bui == "00000": label = 0
                            elif bui.startswith("100"): label = 1 # Bebop
                            elif bui.startswith("101"): label = 2 # AR Drone
                            elif bui.startswith("110"): label = 3 # Phantom
                            else: continue
                            
                            rec_id = re.sub(r'_Seg\d+\.png$', '', f, flags=re.IGNORECASE)
                            self.samples.append({'path': os.path.join(root, f), 'label': label, 'rec_id': rec_id})

    def inject_noise(self, tensor):
        if self.target_snr is None or self.target_snr >= 35: return tensor
        sigma = 1.0 / (10**(self.target_snr / 20))
        return torch.clamp(tensor + torch.randn_like(tensor) * sigma, 0, 1)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['path']).convert('RGB')
        return self.inject_noise(self.transform(img)), item['label']

def get_smart_splits(dataset, train_ratio=0.8):
    # La logique du split par ID reste identique pour éviter le leakage
    rec_to_indices = {}
    for idx, sample in enumerate(dataset.samples):
        rid = sample['rec_id']
        if rid not in rec_to_indices: rec_to_indices[rid] = []
        rec_to_indices[rid].append(idx)
    
    # On groupe par le nouveau label de type (0 à 3)
    class_to_recs = {}
    for rid, indices in rec_to_indices.items():
        label = dataset.samples[indices[0]]['label']
        if label not in class_to_recs: class_to_recs[label] = []
        class_to_recs[label].append(rid)

    train_idx, val_idx = [], []
    random.seed(42)
    for label, recs in class_to_recs.items():
        random.shuffle(recs)
        split = int(len(recs) * train_ratio)
        for rid in recs[:split]: train_idx.extend(rec_to_indices[rid])
        for rid in recs[split:]: val_idx.extend(rec_to_indices[rid])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
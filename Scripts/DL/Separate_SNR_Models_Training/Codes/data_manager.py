import os, re, torch, random
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

class DroneRFDataset(Dataset):
    def __init__(self, folders_list, target_snr=None):
        self.samples = []
        self.target_snr = target_snr
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        # Mapping BUI -> Class ID (0 à 12)
        self.bui_map = {
            "00000": 0, "10000": 1, "10001": 2, "10010": 3, "10011": 4,
            "10100": 5, "10101": 6, "10110": 7, "10111": 8,
            "11000": 9, "11001": 10, "11010": 11, "11011": 12
        }

        print(f"Scan des dossiers pour {target_snr}dB...")
        for folder_path in folders_list:
            if not os.path.exists(folder_path): continue
            for f in os.listdir(folder_path):
                if f.lower().endswith('.png'):
                    # On cherche le code BUI dans le nom de fichier
                    match = re.search(r'([01]{5})', f)
                    if match:
                        bui = match.group(1)
                        label = self.bui_map.get(bui, -1)
                        if label != -1:
                            rec_id = re.sub(r'_Seg\d+\.png$', '', f, flags=re.IGNORECASE)
                            self.samples.append({'path': os.path.join(folder_path, f), 'label': label, 'rec_id': rec_id})

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
    rec_to_indices = {}
    for idx, sample in enumerate(dataset.samples):
        rid = sample['rec_id']
        if rid not in rec_to_indices: rec_to_indices[rid] = []
        rec_to_indices[rid].append(idx)
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
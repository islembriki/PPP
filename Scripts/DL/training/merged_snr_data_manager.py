import os, re, torch, random
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

class DroneRFDatasetMerged(Dataset):
    def __init__(self, folders_list, target_snr=None, bui_map=None):
        self.samples = []
        self.target_snr = target_snr
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        self.bui_map = bui_map if bui_map else {
            "00000": 0, "10000": 1, "10001": 2, "10010": 3, "10011": 4,
            "10100": 5, "10101": 6, "10110": 7, "10111": 8,
            "11000": 9, "11001": 10, "11010": 11, "11011": 12
        }

        for folder_path in folders_list:
            if not os.path.exists(folder_path): continue
            is_background = "Background" in folder_path

            # Get list of files first to show a progress bar for cleaning
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

            for f in tqdm(files, desc=f"Cleaning {os.path.basename(folder_path)}", leave=False):
                match = re.search(r'([01]{5})', f)
                if match:
                    full_path = os.path.join(folder_path, f)

                    # --- SMART PURIFICATION (SAD) ---
                    if not is_background:
                        with Image.open(full_path).convert('L') as temp_img:
                            if np.mean(np.array(temp_img)) < 5:
                                continue
                                # ---------------------------------

                    label = self.bui_map.get(match.group(1), -1)
                    if label != -1:
                        self.samples.append({
                            'path': full_path,
                            'label': label,
                            'rec_id': f.split('_Seg')[0],
                            'snr': target_snr
                        })

    def inject_noise(self, tensor):
        if self.target_snr is None or self.target_snr >= 35:
            return tensor
        sigma = (1.0 / (10**(self.target_snr / 20.0))) * 0.1
        noise = torch.randn_like(tensor) * sigma
        return torch.clamp(tensor + noise, 0, 1)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item['path']).convert('RGB')
        return self.inject_noise(self.transform(img)), item['label']

def get_smart_splits_merged(dataset, train_ratio=0.8):
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
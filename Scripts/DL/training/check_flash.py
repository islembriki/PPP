import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms

# Chemins directs vers tes dossiers
FOLDERS = {
    "Background": r"C:\Users\USER\Desktop\Projet_Drone\data_processed\data_processed\spectrograms\Background activities",
    "Bebop": r"C:\Users\USER\Desktop\Projet_Drone\Bebop_spectorgrams\Bebop",
    "AR Drone": r"C:\Users\USER\Desktop\Projet_Drone\data_processed\data_processed\spectrograms\AR Drone",
    "Phantom": r"C:\Users\USER\Desktop\Projet_Drone\Phantom_spectograms\Phantom"
}

plt.figure(figsize=(16, 4))

for i, (name, path) in enumerate(FOLDERS.items()):
    found = False
    if os.path.exists(path):
        # On liste les fichiers mais on s'arrête au premier PNG
        for f in os.listdir(path):
            if f.lower().endswith('.png'):
                img_path = os.path.join(path, f)
                img = Image.open(img_path).convert('RGB')
                
                plt.subplot(1, 4, i+1)
                plt.imshow(img)
                plt.title(f"TYPE: {name}\n{f[:20]}...", fontsize=10)
                plt.axis('off')
                print(f"✅ OK pour {name}")
                found = True
                break # ON S'ARRÊTE ICI ! Très important pour ne pas ramer.
    
    if not found:
        print(f"❌ Dossier introuvable ou vide : {path}")

plt.tight_layout()
plt.show()
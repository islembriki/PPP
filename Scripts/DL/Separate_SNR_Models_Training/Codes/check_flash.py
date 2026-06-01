import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms

# Chemins directs vers les dossiers
FOLDERS = {
    "Bebop": r"C:\Users\USER\Desktop\Projet_Drone\Bebop\Bebop",
    "Phantom": r"C:\Users\USER\Desktop\Projet_Drone\Phantom\Phantom",
    "AR Drone": r"C:\Users\USER\Desktop\Projet_Drone\AR_Drone\AR_Drone",
    "Background": r"C:\Users\USER\Desktop\Projet_Drone\Background\Background"
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
                print(f"OK pour {name}")
                found = True
                break # ON S'ARRÊTE ICI ! Très important pour ne pas ramer.
    
    if not found:
        print(f"Dossier introuvable ou vide : {path}")

plt.tight_layout()
plt.show()
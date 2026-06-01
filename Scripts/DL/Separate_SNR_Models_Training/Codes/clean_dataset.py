import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Nettoie SEULEMENT les dossiers de drones. 
FOLDERS = [
    r"C:\Users\USER\Desktop\Projet_Drone\Bebop\Bebop",
    r"C:\Users\USER\Desktop\Projet_Drone\Phantom\Phantom",
    r"C:\Users\USER\Desktop\Projet_Drone\AR_Drone\AR_Drone"
]

# SEUIL DE NETTOYAGE : Si la moyenne des pixels est trop basse, c'est du noir.
THRESHOLD = 15 

print("Démarrage du nettoyage des segments vides...")

for folder in FOLDERS:
    if not os.path.exists(folder): continue
    files = [f for f in os.listdir(folder) if f.endswith('.png')]
    removed = 0
    
    for f in tqdm(files, desc=f"Traitement {os.path.basename(folder)}"):
        path = os.path.join(folder, f)
        img = Image.open(path).convert('L') # Ouvrir en niveaux de gris
        arr = np.array(img)
        
        # Si l'image est trop sombre (moyenne des pixels faible)
        if np.mean(arr) < THRESHOLD:
            os.remove(path)
            removed += 1
            
    print(f"{os.path.basename(folder)} : {removed} images vides supprimées.")

print("\nNettoyage terminé ! Ton dataset ne contient maintenant que du VRAI signal.")
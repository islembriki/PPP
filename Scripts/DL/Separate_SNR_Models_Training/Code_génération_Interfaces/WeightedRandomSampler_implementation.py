import matplotlib.pyplot as plt
import numpy as np

# Données basées sur les résultats de nettoyage
classes = ['Background', 'Bebop', 'AR Drone', 'Phantom']
counts_initial = [82000, 19000, 15000, 4000] 

# Calcul des poids (Logique du Sampler)
weights = [1/c for c in counts_initial]
# Normalisation pour l'affichage (probabilité dans un batch équilibré)
batch_dist = [1/len(classes)] * len(classes) 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- GRAPHIQUE 1 : Distribution Réelle ---
ax1.bar(classes, counts_initial, color=['#34495e', '#3498db', '#9b59b6', '#e67e22'])
ax1.set_title("1. Distribution dans le Dataset\n(Déséquilibre après nettoyage)", fontsize=14, fontweight='bold')
ax1.set_ylabel("Nombre d'images (Segments)")
for i, v in enumerate(counts_initial):
    ax1.text(i, v + 1000, str(v), ha='center', fontweight='bold')

# --- GRAPHIQUE 2 : Distribution dans un Batch  ---
ax2.bar(classes, [25, 25, 25, 25], color=['#34495e', '#3498db', '#9b59b6', '#e67e22'])
ax2.set_title("2. Distribution vue par le CNN (par Batch)\n(Équilibrage via WeightedSampler)", fontsize=14, fontweight='bold')
ax2.set_ylabel("Probabilité d'apparition (%)")
ax2.set_ylim(0, 40)
for i in range(4):
    ax2.text(i, 26, "25%", ha='center', fontweight='bold', color='green')

plt.tight_layout()
plt.savefig("weighted_sampling_explanation.png", dpi=300)
plt.show()

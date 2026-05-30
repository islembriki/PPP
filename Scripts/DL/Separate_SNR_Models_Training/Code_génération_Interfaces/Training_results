import matplotlib.pyplot as plt
import numpy as np

# 1. DONNÉES
snr_levels = ["30dB", "10dB", "0dB", "-10dB"]
accuracy = [80.77, 79.88, 74.57, 71.61]

# 2. CONFIGURATION DU STYLE
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'

# Création de la figure (on augmente légèrement la hauteur pour plus d'espace)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_facecolor('#fdfdfd') 

# --- TITRE PRINCIPAL (Positionné très haut : y=0.98) ---
fig.suptitle("DASHBOARD DE PERFORMANCE : MODÈLES EXPERTS SNR\nClassification Fine : 13 Classes BUI", 
             fontsize=22, fontweight='bold', color="#022b55", y=0.98)

# --- GRAPHIQUE 1 : Bar Chart ---
colors = ['#1a936f', '#88d498', '#f3a712', '#db2b39'] 
bars = ax1.bar(snr_levels, accuracy, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

# Titre du graphe avec un padding interne
ax1.set_title("Précision par Expert SNR", fontsize=17, fontweight='bold', pad=25, color="#023161")
ax1.set_ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.set_facecolor('#fcfcfc')

for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height}%', 
             ha='center', va='bottom', fontsize=11, fontweight='bold', color="#010f1c")

# --- GRAPHIQUE 2 : Line Chart ---
ax2.plot(snr_levels, accuracy, marker='o', linestyle='-', linewidth=4, markersize=12, 
         color='#2980b9', markerfacecolor='white', markeredgewidth=3)
ax2.fill_between(snr_levels, accuracy, color='#3498db', alpha=0.1)

# Titre du graphe avec un padding interne
ax2.set_title("Courbe de Résilience au Bruit", fontsize=17, fontweight='bold', pad=25, color="#023161")
ax2.set_ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.set_facecolor('#fcfcfc')

# --- RÉGLAGE DES MARGES (L'ESPACEMENT CRUCIAL) ---
# top=0.75 : Cela force les graphiques à commencer à 75% de la hauteur, 
# laissant 25% de la page blanche en haut pour le titre principal.
plt.subplots_adjust(top=0.75, bottom=0.15, left=0.1, right=0.9, wspace=0.3)

# Sauvegarde
plt.savefig("dashboard_espacement_parfait.png", dpi=300, bbox_inches='tight')
print("Dashboard avec espacement corrigé généré !")
plt.show()
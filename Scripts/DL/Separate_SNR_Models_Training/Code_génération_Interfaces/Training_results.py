import matplotlib.pyplot as plt
import numpy as np

# 1. DONNÉES RÉELLES (CONFORMES AU CSV)
snr_levels = ["30dB", "10dB", "0dB", "-10dB"]
accuracy = [77.82, 79.67, 73.69, 71.0] # Valeurs mises à jour

# 2. CONFIGURATION DU STYLE
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'

# Création de la figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_facecolor('#fdfdfd') 

# --- TITRE PRINCIPAL (Positionné très haut : y=0.98) ---
fig.suptitle("DASHBOARD DE PERFORMANCE RÉEL : MODÈLES EXPERTS SNR\nClassification Fine : 13 Classes BUI", 
             fontsize=22, fontweight='bold', color="#022b55", y=0.98)

# --- GRAPHIQUE 1 : Bar Chart ---
colors = ['#1a936f', '#88d498', '#f3a712', '#db2b39'] 
bars = ax1.bar(snr_levels, accuracy, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

# Titre du graphe avec un padding interne
ax1.set_title("Précision par Expert SNR", fontsize=17, fontweight='bold', pad=25, color="#023161")
ax1.set_ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.set_facecolor('#fcfcfc')

# Ajout des pourcentages exacts au-dessus des barres
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

# --- RÉGLAGE DES MARGES (Évite le chevauchement) ---
plt.subplots_adjust(top=0.75, bottom=0.15, left=0.1, right=0.9, wspace=0.3)

# Sauvegarde haute qualité
plt.savefig("dashboard_officiel_valeurs_csv.png", dpi=300, bbox_inches='tight')
print("Dashboard officiel avec valeurs CSV généré avec succès !")
plt.show()
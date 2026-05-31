import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_cleaning_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Titre
    plt.text(5, 5.5, "SCHÉMA DE PURIFICATION DU DATASET (DATA CLEANING) PAR Signal Activity Detection (SAD)", 
             fontsize=18, fontweight='bold', ha='center', color='#2c3e50')

    # --- ÉTAPE 1 : DATASET BRUT ---
    ax.add_patch(patches.Rectangle((0.5, 3), 2.5, 1.5, color='#ecf0f1', ec='#7f8c8d'))
    plt.text(1.75, 4.6, "DATASET BRUT\n(Post-Segmentation)", fontsize=12, ha='center', fontweight='bold')
    plt.text(1.75, 3.7, "90% Segments Vides\n10% Segments Actifs", fontsize=10, ha='center', color='#e74c3c')
    
    # Symbole : Confusion
    plt.text(1.75, 2.5, "❌ CONFUSION\nImages Noires Identiques\npour des Labels Différents", 
             fontsize=10, ha='center', bbox=dict(facecolor='red', alpha=0.1))

    # --- FLÈCHE 1 ---
    ax.annotate('', xy=(4, 3.75), xytext=(3, 3.75), arrowprops=dict(arrowstyle='->', lw=3, color='#34495e'))

    # --- ÉTAPE 2 : LE FILTRE (ALGORITHME) ---
    ax.add_patch(patches.Circle((5, 3.75), 0.8, color='#3498db', alpha=0.8))
    plt.text(5, 3.7, "ALGORITHME\nCLEANING\n(Seuil µ > 15)", fontsize=11, ha='center', color='white', fontweight='bold')

    # --- FLÈCHE 2 ---
    ax.annotate('', xy=(7, 3.75), xytext=(6, 3.75), arrowprops=dict(arrowstyle='->', lw=3, color='#34495e'))

    # --- ÉTAPE 3 : DATASET PURIFIÉ ---
    ax.add_patch(patches.Rectangle((7.5, 3), 2.2, 1.5, color='#d5f5e3', ec='#27ae60', lw=2))
    plt.text(8.6, 4.6, "DATASET PROPRE\n(Signal Réel)", fontsize=12, ha='center', fontweight='bold')
    plt.text(8.6, 3.7, "100% Information Utile\nSignatures RF Claires", fontsize=10, ha='center', color='#27ae60')

    # Symbole : Succès
    plt.text(8.6, 2.5, "✅ APPRENTISSAGE\nDifférenciation nette\ndes Signatures Radio", 
             fontsize=10, ha='center', bbox=dict(facecolor='green', alpha=0.1))

    # --- RÉSULTATS CHIFFRÉS ---
    results_box = (
        "📈 IMPACT SUR L'ACCURACY :\n"
        "• AVANT : 7% (Modèle aveugle)\n"
        "• APRÈS : 98% (Type) / 80% (BUI)"
    )
    plt.text(5, 1, results_box, ha="center", fontsize=14, fontweight='bold',
             bbox={"facecolor":"#f39c12", "alpha":0.2, "pad":15, "edgecolor":"#e67e22"})

    plt.savefig("schema_cleaning_drone.png", dpi=300, bbox_inches='tight')
    plt.show()

generate_cleaning_diagram()

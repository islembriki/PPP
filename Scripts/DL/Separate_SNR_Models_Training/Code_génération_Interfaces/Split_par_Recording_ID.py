import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_leakage_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Configuration commune
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')

    # --- CAS A : SPLIT ALÉATOIRE  ---
    ax1.text(5, 4.5, "SPLIT ALÉATOIRE (Data Leakage)", fontsize=14, fontweight='bold', ha='center', color='red')
    # Dessin des segments mélangés
    for i in range(8):
        color = 'blue' if i % 2 == 0 else 'red'
        ax1.add_patch(patches.Rectangle((1+i, 2), 0.8, 1, color=color, alpha=0.6))
    ax1.text(5, 1.5, "Les segments d'un même fichier sont\ndispersés entre Train (Bleu) et Test (Rouge).\nLe modèle mémorise le bruit de fond.", ha='center', fontsize=11)

    # --- CAS B : SPLIT PAR ID  ---
    ax2.text(5, 4.5, "SPLIT PAR RECORDING ID (Généralisation)", fontsize=14, fontweight='bold', ha='center', color='green')
    # Dessin des blocs groupés
    ax2.add_patch(patches.Rectangle((1, 2), 3.5, 1, color='blue', alpha=0.6, label='Train'))
    ax2.add_patch(patches.Rectangle((6, 2), 2, 1, color='red', alpha=0.6, label='Test'))
    ax2.text(2.75, 2.4, "Record A (Complet)", ha='center', color='white', fontweight='bold')
    ax2.text(7, 2.4, "Record B\n(Complet)", ha='center', color='white', fontweight='bold')
    ax2.text(5, 1.5, "Les enregistrements sont isolés.\nLe modèle est forcé d'apprendre\nla signature réelle du drone.", ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig("data_leakage_explanation.png", dpi=300)
    plt.show()

generate_leakage_diagram()

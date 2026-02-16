import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================
# Définition des axes NV
# ================================
NV_AXES = np.array([
    [ 1,  1,  1],   # NV1
    [ 1, -1, -1],   # NV2
    [-1,  1, -1],   # NV3
    [-1, -1,  1]    # NV4
])
NV_AXES = NV_AXES / np.linalg.norm(NV_AXES, axis=1, keepdims=True)

def compute_alignment_with_nv_axes(B_field):
    B_norm = B_field / (np.linalg.norm(B_field) + 1e-12)
    alignments = np.abs(np.dot(NV_AXES, B_norm))
    return alignments, alignments.max()

# ================================
# Génération de champs isotropiques
# ================================
def sample_isotropic_B(n_samples, Bmin=0.5e-3, Bmax=5e-3):
    # Vecteurs unitaires isotropiques
    u = np.random.normal(size=(n_samples,3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    # Rayons uniformes en magnitude
    r = np.random.uniform(Bmin, Bmax, size=(n_samples,1))
    return u * r

def filter_by_alignment(Bs, max_align=0.85):
    # Garde uniquement les B "informative" (non alignés)
    keep = []
    for B in Bs:
        _, a = compute_alignment_with_nv_axes(B)
        if a < max_align:
            keep.append(B)
    return np.array(keep)

# ================================
# Cas alignés et partiellement alignés
# ================================
def generate_nv_aligned_cases(B_magnitudes):
    B_configs = []
    for mag in B_magnitudes:
        for nv_axis in NV_AXES:
            B_configs.append(mag * nv_axis)
    return np.array(B_configs)

def generate_partially_aligned_cases(B_magnitudes, n_angles=3):
    B_configs = []
    for mag in B_magnitudes:
        for nv_axis in NV_AXES:
            # Créer vecteur perpendiculaire
            perp = np.cross(nv_axis, [1,0,0]) if abs(nv_axis[0])<0.9 else np.cross(nv_axis, [0,1,0])
            perp /= np.linalg.norm(perp)
            # Rotation partielle
            for angle in np.linspace(0, np.pi/4, n_angles):  # max 45°
                B = mag * (np.cos(angle) * nv_axis + np.sin(angle) * perp)
                B_configs.append(B)
    return np.array(B_configs)

# ================================
# Générateur final optimisé
# ================================
def generate_optimized_B_dataset(n_total=5000):
    n_main = int(n_total * 0.85)   # 85 % isotrope non aligné
    n_aligned = int(n_total * 0.10)  # 10 % alignés pour robustesse
    n_partial = n_total - n_main - n_aligned  # 5 % partiels

    # 1. Isotropique non aligné
    B_main = sample_isotropic_B(n_main)
    B_main = filter_by_alignment(B_main, max_align=0.85)

    # 2. Alignés
    mags_aligned = np.linspace(1e-3, 4e-3, max(1, n_aligned//4))
    B_aligned = generate_nv_aligned_cases(mags_aligned)
    B_aligned = B_aligned[:n_aligned]

    # 3. Partiellement alignés
    mags_partial = np.linspace(1e-3, 4e-3, max(1, n_partial//(4*3)))
    B_partial = generate_partially_aligned_cases(mags_partial, n_angles=3)

    # Combiner et mélanger
    B_all = np.vstack([B_main, B_aligned, B_partial])
    labels = np.concatenate([
        np.zeros(len(B_main), dtype=int),
        np.ones(len(B_aligned), dtype=int),
        2*np.ones(len(B_partial), dtype=int)
    ])
    idx = np.random.permutation(len(B_all))
    return B_all[idx], labels[idx]

# ================================
# Visualisation
# ================================
def visualize_B_configs(B_configs, labels=None, title="Dataset Optimisé B"):
    fig = plt.figure(figsize=(16,6))
    ax1 = fig.add_subplot(131, projection='3d')
    if labels is not None:
        colors = ['blue', 'red', 'green']
        names = ['Isotropic', 'Aligned', 'Partial']
        for i in range(3):
            mask = labels==i
            ax1.scatter(B_configs[mask,0]*1000, B_configs[mask,1]*1000, B_configs[mask,2]*1000,
                        c=colors[i], label=names[i], s=20, alpha=0.6)
    else:
        ax1.scatter(B_configs[:,0]*1000, B_configs[:,1]*1000, B_configs[:,2]*1000,
                    c='blue', s=20, alpha=0.6)
    ax1.set_xlabel('Bx (mT)'); ax1.set_ylabel('By (mT)'); ax1.set_zlabel('Bz (mT)')
    ax1.set_title("Distribution 3D"); ax1.legend(fontsize=8)

    # Histogramme des magnitudes
    ax2 = fig.add_subplot(132)
    mags = np.linalg.norm(B_configs, axis=1)*1000
    ax2.hist(mags, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('|B| (mT)'); ax2.set_ylabel('N configs')
    ax2.set_title(f'Magnitudes |B| - Moy: {mags.mean():.2f} mT'); ax2.grid(True, alpha=0.3)

    # Histogramme de l'alignement max
    ax3 = fig.add_subplot(133)
    align_max = np.array([compute_alignment_with_nv_axes(B)[1] for B in B_configs])
    ax3.hist(align_max, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax3.set_xlabel('Alignement max axes NV'); ax3.set_ylabel('N configs')
    ax3.set_title('Alignement max'); ax3.grid(True, alpha=0.3)
    ax3.axvline(0.8, color='red', linestyle='--', label='Seuil non aligné')
    ax3.legend(fontsize=8)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

    print("\nSTATISTIQUES DU DATASET")
    print("="*60)
    print(f"Total configs: {len(B_configs)}")
    print(f"Magnitudes |B|: min {mags.min():.3f} mT, max {mags.max():.3f} mT, moy {mags.mean():.3f} mT")
    print(f"Alignement max axes NV: >0.8 {np.sum(align_max>0.8)} ({np.sum(align_max>0.8)/len(align_max)*100:.1f}%)")

# ================================
# MAIN
# ================================
if __name__ == "__main__":
    B_configs, labels = generate_optimized_B_dataset(n_total=10000)
    visualize_B_configs(B_configs, labels)
    np.save('B_configs.npy', B_configs)
    np.save('B_labels.npy', labels)
    print("Dataset sauvegardé: B_configs.npy, B_labels.npy")

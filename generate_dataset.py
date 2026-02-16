import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from Ensemble_Ellipticity_ESR_Simulator_V2 import Ensemble_Spectrum, Ensemble_Spectrum_HF

# ========== CONFIGURATION DU DATASET ==========
N_MW_CONFIGS = 5     # Nombre de configurations MW différentes pour chaque B
FREQ_START = 2.70e3  # MHz
FREQ_END = 3.05e3    # MHz
FREQ_POINTS = 250    # Nombre de points de fréquence
TILT_X = 4.0         # Inclinaison du diamant en degrés (axe x)
TILT_Y = 0.0         # Inclinaison du diamant en degrés (axe y)
INCLUDE_HYPERFINE = True
# OUTPUT_DIR = "odmr_synthetic_dataset"   # Dossier de sortie pour les fichiers générés
OUTPUT_DIR = "test_V2"
B_CONFIGS_FILE = "B_configs.npy"  # Fichier de B configs isotropes

# Configurations MW used for the dataset odmr_synthetic_dataset
# MW_CONFIGS = [
#     ((1.0, 0.2, 0.0), np.deg2rad(180)), # Config 1 : Quasi-linéaire X
#     ((0.2, 1.0, 0.0), np.deg2rad(180)), # Config 2 : Quasi-linéaire Y
#     ((0.2, 0.6, 0.0), np.deg2rad(180)), # Config 3 : Elliptique modéré
#     ((0.6, 0.2, 0.0), np.deg2rad(180)), # Config 4 : Elliptique modéré inversé
#     ((0.8, 0.4, 0.0), np.deg2rad(180)), # Config 5 : Elliptique intermédiaire
# ]

# Configurations MW used for the dataset test
# MW_CONFIGS = [
#     # ((1.0, 0.2, 0.0), 0.0),             # decrease pics 3 and 7
#     # ((1.0, 0.2, 0.0), np.deg2rad(180)), # decrease pics 2 and 6
#     ((0.2, 0.6, 0.0), np.deg2rad(180)), 
#     ((0.6, 0.2, 0.0), np.deg2rad(180)),   # decrease pics 2, 4 and 6

#     ((0.3, 0.7, 0.0), np.deg2rad(60)),    # Neutral

#     ((0.8, 0.4, 0.0), 0.0),               # decrease pics 3 and 5
#     ((0.8, 0.4, 0.0), np.deg2rad(180)),   # decrease pics 4 and 6
# ]

# MW Configurations used for the dataset test_V2
MW_CONFIGS = [
    # ((1.0, 0.1, 0.0), np.deg2rad(90)), # neutre

    # ((0.6, 0.4, 0.0), np.deg2rad(90)),    # ↓peaks 4, 6
    # ((0.6, 0.4, 0.0), np.deg2rad(225.0)), # ↓peaks 1, 7
    # ((0.3, 0.7, 0.0), np.deg2rad(60)),    # ↓peaks 3, 5 (perfect)
    # ((0.2, 0.6, 0.0), np.deg2rad(180)),   # ↓peaks 1, 3, 5, 7
    # ((0.7, 0.4, 0.0), np.deg2rad(180)),   # ↓peaks 2, 8

    # ((1.5, 0.5, 0.0), np.deg2rad(90)), # ↓peak 5
    # ((1.0, 1.0, 0.0), np.deg2rad(90)), # near-neutral
    # ((0.7, 0.7, 0.0), np.deg2rad(90)),
    # ((0.5, 0.5, 0.0), np.deg2rad(90)),
    # ((0.2, 0.2, 0.0), np.deg2rad(90)),
    # ((2.0, 0.0, 0.0), np.deg2rad(90)),

    ((0.80, 0.70, 0.0), np.deg2rad(120.0)),
    ((0.20, 0.60, 0.0), np.deg2rad(60.0)),
    ((0.20, 0.70, 0.0), np.deg2rad(180.0)),
    ((1.00, 0.60, 0.0), np.deg2rad(180.0)),
    ((0.70, 1.60, 0.0), np.deg2rad(300.0)),    
]

# Nombre de processus parallèles (utilise tous les CPU disponibles par défaut)
N_PROCESSES = cpu_count()


def compute_spectra_for_B_config(args):
    """
    Calcule tous les spectres MW pour une configuration B donnée.
    
    Parameters:
        args : tuple
            (b_idx, B_field, mw_configs, freq_list, tilt_x, tilt_y, include_hyperfine)        
    Returns:
        b_idx : int
            Index de configuration B
        all_spectra : array
            Spectres ODMR (shape: N_MW_CONFIGS x FREQ_POINTS)
        B_field : list
            Configuration du champ magnétique
    """
    b_idx, B_field, mw_configs, freq_list, tilt_x, tilt_y = args
    
    n_mw_configs = len(mw_configs)
    freq_points = len(freq_list)
    all_spectra = np.zeros((n_mw_configs, freq_points))
    
    for mw_idx, mw_config in enumerate(mw_configs):
        MW_field, MW_phase = mw_config  # mw_config = ((Ox, Oy, Oz), phase)
        MW_field = list(MW_field)  # Convertir tuple en liste
        
        odmr_spectrum = Ensemble_Spectrum(
            B_field, 
            MW_field, 
            MW_phase, 
            freq_list, 
            tilt_x=tilt_x, 
            tilt_y=tilt_y, 
        )
        
        all_spectra[mw_idx, :] = odmr_spectrum
    
    return b_idx, all_spectra, B_field


def generate_dataset(output_dir=OUTPUT_DIR, n_mw_configs=N_MW_CONFIGS, n_processes=N_PROCESSES):
    """
    Génère le dataset complet de spectres ODMR en parallèle.
    
    Parameters:
        output_dir : str
            Dossier de sortie
        n_mw_configs : int
            Nombre de configurations MW par champ magnétique
        n_processes : int
            Nombre de processus parallèles
    """    
    # Création du dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "signals"), exist_ok=True)

    # Utilisation des configurations MW prédéfinies
    mw_configs = MW_CONFIGS[:n_mw_configs]
    
    # Sauvegarder les configurations MW dans un fichier numpy
    mw_configs_array = np.array([(list(mw_field) + [phase]) for mw_field, phase in mw_configs])
    mw_configs_file = os.path.join(output_dir, "mw_configs.npy")
    np.save(mw_configs_file, mw_configs_array)
    
    # Chargement des configurations de champ magnétique (balayage isotrope)
    B_configs_all = np.load(B_CONFIGS_FILE).tolist()
    # B_configs = [B_configs_all[0]]
    B_configs = B_configs_all
    n_b_configs = len(B_configs)
    
    # Génération de la liste des fréquences
    freq_list = np.linspace(FREQ_START, FREQ_END, FREQ_POINTS)

    # Enregistrement des fréquences
    freq_file = os.path.join(output_dir, "frequencies.npy")
    np.save(freq_file, freq_list)
    
    print(f"  B configurations (nbr of files): {n_b_configs}")
    print(f"  MW configurations (spectrum per file) : {n_mw_configs}")
    print(f"  Total spectrum to calculate: {n_b_configs * n_mw_configs}")

    # Préparer les arguments pour chaque configuration B
    args_list = [
        (b_idx, B_field, mw_configs, freq_list, TILT_X, TILT_Y)
        for b_idx, B_field in enumerate(B_configs)
    ]
    
    # Calcul parallèle avec multiprocessing
    configs_data = []
    
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(compute_spectra_for_B_config, args_list),
            total=len(args_list),
            desc="Generation in progress"
        ))
    
    # Save results
    for b_idx, all_spectra, B_field in tqdm(results, desc="Saving"):
        Bx, By, Bz = B_field
        
        # Enregistrement du fichier signal
        signal_file = os.path.join(output_dir, "signals", f"config_{b_idx:05d}.npy")
        np.save(signal_file, all_spectra)
        
        # Ajout à la configuration
        configs_data.append({
            'experiment_id': b_idx,
            'Bx': Bx,
            'By': By,
            'Bz': Bz
        })
    
    # Sauvegarde du fichier de configurations
    configs_df = pd.DataFrame(configs_data)
    configs_file = os.path.join(output_dir, "configs.csv")
    configs_df.to_csv(configs_file, index=False)

    print("\n✓ Dataset generation completed, saved in:", output_dir)


if __name__ == "__main__":
    generate_dataset()

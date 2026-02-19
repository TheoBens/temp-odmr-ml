import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# ========== CONFIGURATION ==========
OUTPUT_DIR = "datasets/odmr_synthetic_dataset_2"

def load_dataset():
    """
    Charge le dataset complet depuis les fichiers.
    
    Returns:
        configs_df : DataFrame
            Configurations des champs magnétiques
        mw_configs : array
            Configurations MW (Ox, Oy, Oz, phase)
        frequencies : array
            Liste des fréquences
        n_b_configs : int
            Nombre de configurations B
        n_mw_configs : int
            Nombre de configurations MW
    """
    # Charger les configurations B
    configs_file = os.path.join(OUTPUT_DIR, "configs.csv")
    configs_df = pd.read_csv(configs_file)
    n_b_configs = len(configs_df)
    
    # Charger les configurations MW
    mw_configs_file = os.path.join(OUTPUT_DIR, "mw_configs.npy")
    mw_configs = np.load(mw_configs_file)
    n_mw_configs = len(mw_configs)
    
    # Charger les fréquences
    freq_file = os.path.join(OUTPUT_DIR, "frequencies.npy")
    frequencies = np.load(freq_file)
    
    return configs_df, mw_configs, frequencies, n_b_configs, n_mw_configs


def load_spectrum(b_idx, mw_idx):
    """
    Charge un spectre spécifique depuis les fichiers.
    
    Parameters:
        b_idx : int
            Index de la configuration B
        mw_idx : int
            Index de la configuration MW
            
    Returns:
        spectrum : array
            Spectre ODMR
    """
    signal_file = os.path.join(OUTPUT_DIR, "signals", f"config_{b_idx:05d}.npy")
    all_spectra = np.load(signal_file)
    return all_spectra[mw_idx, :]


def visualize_dataset():
    """
    Visualise le dataset avec des sliders interactifs pour naviguer
    entre les différentes configurations B et MW.
    """
    # Charger le dataset
    configs_df, mw_configs, frequencies, n_b_configs, n_mw_configs = load_dataset()
    
    # Créer la figure et les axes
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.90)
    
    # Charger le premier spectre
    spectrum = load_spectrum(0, 0)
    
    # Tracer le spectre initial
    line, = ax.plot(frequencies, spectrum, 'b-', linewidth=1.5)
    ax.set_xlabel('Fréquence (MHz)', fontsize=12)
    ax.set_ylabel('Signal ODMR (u.a.)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Titre avec les informations de configuration
    B_config = configs_df.iloc[0]
    MW_config = mw_configs[0]
    title_text = (f'Configuration B #{0}: Bx={B_config["Bx"]:.2f}G, By={B_config["By"]:.2f}G, Bz={B_config["Bz"]:.2f}G\n'
                  f'Configuration MW #{0}: Ox={MW_config[0]:.2f}, Oy={MW_config[1]:.2f}, Oz={MW_config[2]:.2f}, Phase={np.rad2deg(MW_config[3]):.1f}°')
    title = ax.set_title(title_text, fontsize=11, pad=10)
    
    # Créer les axes pour les sliders
    ax_b_slider = plt.axes([0.1, 0.12, 0.8, 0.03])
    ax_mw_slider = plt.axes([0.1, 0.06, 0.8, 0.03])
    
    # Créer les sliders
    b_slider = Slider(
        ax=ax_b_slider,
        label='Config B',
        valmin=0,
        valmax=n_b_configs - 1,
        valinit=0,
        valstep=1,
        color='steelblue'
    )
    
    mw_slider = Slider(
        ax=ax_mw_slider,
        label='Config MW',
        valmin=0,
        valmax=n_mw_configs - 1,
        valinit=0,
        valstep=1,
        color='coral'
    )
    
    # Fonction de mise à jour
    def update(val):
        b_idx = int(b_slider.val)
        mw_idx = int(mw_slider.val)
        
        # Charger le nouveau spectre
        spectrum = load_spectrum(b_idx, mw_idx)
        
        # Mettre à jour le tracé
        line.set_ydata(spectrum)
        
        # Mettre à jour le titre
        B_config = configs_df.iloc[b_idx]
        MW_config = mw_configs[mw_idx]
        new_title = (f'Configuration B #{b_idx}: Bx={B_config["Bx"]:.2f}G, By={B_config["By"]:.2f}G, Bz={B_config["Bz"]:.2f}G\n'
                     f'Configuration MW #{mw_idx}: Ox={MW_config[0]:.2f}, Oy={MW_config[1]:.2f}, Oz={MW_config[2]:.2f}, Phase={np.rad2deg(MW_config[3]):.1f}°')
        title.set_text(new_title)
        
        # Ajuster l'échelle Y pour mieux voir le signal
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        
        fig.canvas.draw_idle()
    
    # Connecter les sliders à la fonction de mise à jour
    b_slider.on_changed(update)
    mw_slider.on_changed(update)
    
    # Afficher le graphique
    plt.show()


if __name__ == "__main__":
    visualize_dataset()

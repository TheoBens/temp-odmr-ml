import os
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Ensemble_Ellipticity_ESR_Simulator import Ensemble_Spectrum

'''
Plot all spectra for all MW configurations for a given B field, to determine the optimal MW configurations.
'''

N_MW_CONFIGS = 5     # Number of different MW configurations
FREQ_START = 2.70e3  # MHz
FREQ_END = 3.05e3    # MHz
FREQ_POINTS = 250    # Number of frequency points
TILT_X = 0.0         # Diamond tilt in degrees (x-axis)
TILT_Y = 0.0         # Diamond tilt in degrees (y-axis)
MW_X_SWEEP = np.linspace(0.1, 2.0, 20)
MW_Y_SWEEP = np.linspace(0.1, 2.0, 20)
MW_PHASES = np.array([0, 60, 120, 180, 240, 300])
B = [0.005, 0.002, 0.003]
freq_list = np.linspace(FREQ_START, FREQ_END, FREQ_POINTS)

def get_all_spectra(B, mw_x_sweep, mw_y_sweep, mw_phases, tilt_x_deg, tilt_y_deg):
    """
    Compute ODMR spectra for all MW configurations on a grid of
    MW amplitudes (Ox, Oy) and relative phase.

    Parameters
        B : array-like, shape (3,)
            Static magnetic field [Bx, By, Bz] in Tesla.
        mw_x_sweep, mw_y_sweep : 1D arrays
            Ranges of MW amplitudes along x and y (arb. units).
        mw_phases : 1D array
            MW phases in degrees.
        tilt_x_deg, tilt_y_deg : float
            Diamond tilt angles in degrees.
    Returns
        configs : ndarray, shape (n_configs, 3)
            Each row is (Ox, Oy, phase_deg) describing one MW configuration.
        all_spectra : ndarray, shape (n_configs, FREQ_POINTS)
            Simulated ODMR spectrum for each configuration.
    """
    mw_x_sweep = np.asarray(mw_x_sweep)
    mw_y_sweep = np.asarray(mw_y_sweep)
    mw_phases = np.asarray(mw_phases)

    n_mw_configs = len(mw_x_sweep) * len(mw_y_sweep) * len(mw_phases)
    all_spectra = np.zeros((n_mw_configs, FREQ_POINTS))
    configs = np.zeros((n_mw_configs, 3))

    # Convert tilts and phases to radians for the simulator
    tilt_x = np.deg2rad(tilt_x_deg)
    tilt_y = np.deg2rad(tilt_y_deg)

    idx = 0
    # Iterate over all MW configurations with a progress bar
    for ox, oy, phase_deg in tqdm(
        product(mw_x_sweep, mw_y_sweep, mw_phases),
        total=n_mw_configs,
        desc="Simulating MW configurations",
    ):
        phase_rad = np.deg2rad(phase_deg)

        MW_field = [ox, oy, 0.0]
        spectrum = Ensemble_Spectrum(B, MW_field, phase_rad, freq_list, tilt_x, tilt_y)

        all_spectra[idx, :] = spectrum
        configs[idx, :] = [ox, oy, phase_deg]
        idx += 1

    return configs, all_spectra


def plot_all_spectra(freq_list, configs, all_spectra, output_dir="mw_spectra_figs", n_per_fig=8):
    """
    Save figures with 8 spectra per figure in a dedicated folder.

    Each subplot corresponds to one MW configuration.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_configs = all_spectra.shape[0]
    n_figs = int(np.ceil(n_configs / n_per_fig))

    for fig_idx in tqdm(range(n_figs), desc="Saving figures"):
        start = fig_idx * n_per_fig
        end = min(start + n_per_fig, n_configs)
        n_this = end - start

        fig, axes = plt.subplots(n_this, 1, figsize=(8, 2.5 * n_this), sharex=True)
        if n_this == 1:
            axes = [axes]

        for ax, cfg_idx in zip(axes, range(start, end)):
            spectrum = all_spectra[cfg_idx]
            ox, oy, phase_deg = configs[cfg_idx]

            ax.plot(freq_list, spectrum, "b-")
            ax.set_ylabel("Fluor. (a.u.)")
            ax.set_title(
                f"Config {cfg_idx}: Ox={ox:.2f}, Oy={oy:.2f}, phase={phase_deg:.0f}°"
            )
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Frequency (MHz)")
        fig.tight_layout()

        save_path = os.path.join(output_dir, f"mw_spectra_{fig_idx:03d}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {n_figs} figure(s) in '{output_dir}'.")


def _simulate_spectrum_for_B_and_config(args):
    """
    Worker function for multiprocessing: simulate one spectrum
    for a given (B_field, MW config).

    Parameters
    ----------
    args : tuple
        (B_field, ox, oy, phase_deg, tilt_x, tilt_y)
    """
    B_field, ox, oy, phase_deg, tilt_x, tilt_y = args
    phase_rad = np.deg2rad(phase_deg)
    MW_field = [ox, oy, 0.0]
    spectrum = Ensemble_Spectrum(
        B_field, MW_field, phase_rad, freq_list, tilt_x, tilt_y
    )
    return spectrum


def plot_selected_configs_vs_B_fields(
    output_path="custom_mw_configs_vs_B.png",
    n_processes=None,
):
    """
    Plot a set of hand-picked MW configurations for several magnetic fields
    on a single figure (rows = B fields, columns = MW configs).
    """
    # Hand-picked MW configurations: (Ox, Oy, phase_deg)
    custom_configs = [
        (0.80, 0.70, 120.0),
        (0.20, 0.60, 60.0),
        (0.20, 0.70, 180.0),
        (1.00, 0.60, 180.0),
        (0.70, 1.60, 300.0),
    ]

    # A few representative magnetic field configurations (Tesla)
    B_fields = [
        [0.005, 0.002, 0.003],
        [0.004, 0.001, 0.003],
        [0.006, 0.003, 0.003],
        [0.005, -0.001, 0.003],
        [0.005, 0.002, 0.004],
    ]

    tilt_x = np.deg2rad(TILT_X)
    tilt_y = np.deg2rad(TILT_Y)

    n_B = len(B_fields)
    n_cfg = len(custom_configs)

    # --------- Parallel simulation over (B, MW_config) pairs ---------
    jobs = []
    for B_field in B_fields:
        for (ox, oy, phase_deg) in custom_configs:
            jobs.append((B_field, ox, oy, phase_deg, tilt_x, tilt_y))

    if n_processes is None:
        n_processes = min(cpu_count(), len(jobs))

    with Pool(processes=n_processes) as pool:
        spectra_list = list(
            tqdm(
                pool.imap(_simulate_spectrum_for_B_and_config, jobs),
                total=len(jobs),
                desc="Simulating spectra (B, MW config)",
            )
        )

    spectra = np.array(spectra_list).reshape(n_B, n_cfg, FREQ_POINTS)

    # --------- Plotting ---------
    fig, axes = plt.subplots(
        n_B,
        n_cfg,
        figsize=(3.0 * n_cfg, 2.5 * n_B),
        sharex=True,
        sharey=True,
    )

    # Ensure axes is 2D array-like
    if n_B == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cfg == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, B_field in enumerate(B_fields):
        for j, (ox, oy, phase_deg) in enumerate(custom_configs):
            ax = axes[i][j]
            spectrum = spectra[i, j]

            ax.plot(freq_list, spectrum, "b-")

            if i == n_B - 1:
                ax.set_xlabel("Frequency (MHz)")
            if j == 0:
                bx, by, bz = B_field
                ax.set_ylabel(f"B=({bx:.3f},{by:.3f},{bz:.3f}) T\nFluor. (a.u.)")

            if i == 0:
                ax.set_title(
                    f"Ox={ox:.2f}, Oy={oy:.2f}, φ={phase_deg:.0f}°",
                    fontsize=9,
                )

            ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Selected MW configurations for different magnetic fields",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved custom configs vs B figure to '{output_path}'.")


def main():
    print("Computing spectra for all MW configurations...")
    configs, spectra = get_all_spectra(
        B=B,
        mw_x_sweep=MW_X_SWEEP,
        mw_y_sweep=MW_Y_SWEEP,
        mw_phases=MW_PHASES,
        tilt_x_deg=TILT_X,
        tilt_y_deg=TILT_Y,
    )
    print(f"Computed {spectra.shape[0]} spectra "
          f"over {spectra.shape[1]} frequency points.")

    # Save configs (and optionally spectra) alongside figures
    output_dir = "mw_spectra_figs_without_tilt"
    os.makedirs(output_dir, exist_ok=True)

    # Save as .npy for exact reuse
    np.save(os.path.join(output_dir, "mw_configs.npy"), configs)
    np.save(os.path.join(output_dir, "mw_spectra.npy"), spectra)

    # Also save a human‑readable CSV for quick inspection
    csv_path = os.path.join(output_dir, "mw_configs.csv")
    header = "index,Ox,Oy,phase_deg"
    indices = np.arange(configs.shape[0]).reshape(-1, 1)
    np.savetxt(
        csv_path,
        np.hstack([indices, configs]),
        delimiter=",",
        header=header,
        comments="",
    )
    print(f"Saved MW configs and spectra in '{output_dir}'.")

    plot_all_spectra(freq_list, configs, spectra, output_dir=output_dir)

    # Also generate comparison figure for the requested specific configs
    # plot_selected_configs_vs_B_fields(
    #     output_path=os.path.join(output_dir, "custom_mw_configs_vs_B.png")
    # )


if __name__ == "__main__":
    main()

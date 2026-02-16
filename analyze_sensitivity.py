import numpy as np
from tqdm import tqdm

from Ensemble_Ellipticity_ESR_Simulator_V2 import Ensemble_Spectrum
from generate_dataset import (
    MW_CONFIGS,
    FREQ_START,
    FREQ_END,
    FREQ_POINTS,
    TILT_X,
    TILT_Y,
    B_CONFIGS_FILE,
)


def compute_sensitivity_for_B(B_field, delta=1e-4):
    """
    Calcule une mesure simple de sensibilité des spectres ODMR à Bx, By, Bz.

    Pour chaque composante i, on approxime numériquement :
        dSpectrum/dB_i ≈ (S(B_i + delta) - S(B_i - delta)) / (2 * delta)

    et on renvoie la norme L2 moyenne de cette dérivée sur les MW configs.
    """
    freq_list = np.linspace(FREQ_START, FREQ_END, FREQ_POINTS)
    tilt_x = TILT_X * np.pi / 180.0
    tilt_y = TILT_Y * np.pi / 180.0

    B_field = np.array(B_field, dtype=float)

    sensitivities = []
    for i, comp_name in enumerate(["Bx", "By", "Bz"]):
        B_plus = B_field.copy()
        B_minus = B_field.copy()
        B_plus[i] += delta
        B_minus[i] -= delta

        deriv_norms = []
        for MW_field, MW_phase in MW_CONFIGS:
            MW_field = list(MW_field)

            S_plus = Ensemble_Spectrum(B_plus, MW_field, MW_phase, freq_list, tilt_x, tilt_y)
            S_minus = Ensemble_Spectrum(B_minus, MW_field, MW_phase, freq_list, tilt_x, tilt_y)

            dS = (S_plus - S_minus) / (2 * delta)
            deriv_norms.append(np.linalg.norm(dS))

        deriv_norms = np.array(deriv_norms)
        sensitivities.append(
            {
                "component": comp_name,
                "mean_norm": float(deriv_norms.mean()),
                "std_norm": float(deriv_norms.std()),
            }
        )

    return sensitivities


def main(n_samples=5, delta=1e-4):
    # Charge quelques configurations B déjà utilisées pour le dataset
    B_configs_all = np.load(B_CONFIGS_FILE).tolist()
    if len(B_configs_all) == 0:
        raise RuntimeError("B_CONFIGS_FILE is empty.")

    indices = np.linspace(0, len(B_configs_all) - 1, num=n_samples, dtype=int)
    print(f"Analyzing sensitivity on {n_samples} B-field samples (indices: {indices.tolist()})")

    all_results = []
    for idx in tqdm(indices, desc="Sensitivity over B samples"):
        B_field = B_configs_all[idx]
        sens = compute_sensitivity_for_B(B_field, delta=delta)
        all_results.append((idx, B_field, sens))

    print("\n================ SENSITIVITY SUMMARY ================")
    for idx, B_field, sens in all_results:
        print(f"\nB index {idx}, B = {B_field}")
        for s in sens:
            print(
                f"  dSpectrum/d{s['component']}: "
                f"mean ||·|| = {s['mean_norm']:.4e}, std = {s['std_norm']:.4e}"
            )


if __name__ == "__main__":
    main()


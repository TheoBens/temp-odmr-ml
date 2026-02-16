"""
Pipeline d'optimisation automatique pour trouver la meilleure configuration
de dataset (configs MW) et de modèle pour la prédiction de champ magnétique.

Cette pipeline :
1. Génère un dataset avec des configs MW optimisées
2. Entraîne plusieurs modèles prometteurs
3. Évalue et sauvegarde le meilleur modèle
4. Itère pour améliorer continuellement les résultats
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

from Ensemble_Ellipticity_ESR_Simulator_V2 import Ensemble_Spectrum
from dataloader import load_odmr_dataset, create_train_test_split, create_dataloaders
from models import get_model, count_parameters


# ===================================
# CONFIGURATION GLOBALE
# ===================================
N_ITERATIONS = 50               # Nombre d'itérations d'optimisation
N_B_CONFIGS = 5000              # Nombre de configs de champ magnétique
N_MW_CONFIGS = 5                # Nombre de configs MW par dataset
FREQ_START = 2.70e3             # MHz
FREQ_END = 3.05e3               # MHz
FREQ_POINTS = 250               # Nombre de points de fréquence
TILT_X = 4.0                    # Inclinaison (degrés)
TILT_Y = 0.0
INCLUDE_HYPERFINE = True

# Modèles à tester
MODELS_TO_TEST = ["densenet", "cnn1d"]

# Configs d'entraînement
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_EPOCHS = 150
PATIENCE = 30

# Dossiers
BEST_RESULTS_DIR = "best_optimization_results"
TEMP_DATASET_DIR = "temp_optimization_dataset"
B_CONFIGS_FILE = "B_configs.npy"

# Métrique d'optimisation
OPTIMIZATION_METRIC = "avg_rmse"  # RMSE moyen sur Bx, By, Bz


# ===================================
# STRATÉGIES DE GÉNÉRATION MW
# ===================================

def generate_mw_configs_strategy(strategy_name, n_configs=5, seed=None):
    """
    Génère des configs MW selon différentes stratégies.
    
    Returns:
        mw_configs : list of tuples
            [(MW_field, MW_phase), ...]
            MW_field = (Ox, Oy, Oz)
            MW_phase = phase in radians
    """
    if seed is not None:
        np.random.seed(seed)
    
    configs = []
    
    if strategy_name == "random":
        # Totalement aléatoire
        for _ in range(n_configs):
            ox = np.random.uniform(0.1, 1.5)
            oy = np.random.uniform(0.1, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            configs.append(((ox, oy, 0.0), phase))
    
    elif strategy_name == "focused":
        # Configs ciblées autour de zones prometteuses
        base_configs = [
            (0.8, 0.7, 120),
            (0.2, 0.6, 60),
            (0.2, 0.7, 180),
            (1.0, 0.6, 180),
            (0.7, 1.6, 300),
        ]
        for ox, oy, phase_deg in base_configs[:n_configs]:
            # Ajouter perturbation
            ox += np.random.normal(0, 0.1)
            oy += np.random.normal(0, 0.1)
            phase_deg += np.random.normal(0, 20)
            
            ox = np.clip(ox, 0.1, 2.0)
            oy = np.clip(oy, 0.1, 2.0)
            phase = np.deg2rad(phase_deg % 360)
            
            configs.append(((ox, oy, 0.0), phase))
    
    elif strategy_name == "ellipticity_sweep":
        # Sweep sur différentes ellipticités
        phases = np.linspace(0, 2*np.pi, n_configs, endpoint=False)
        for phase in phases:
            ox = np.random.uniform(0.3, 1.2)
            oy = np.random.uniform(0.3, 1.2)
            configs.append(((ox, oy, 0.0), phase))
    
    elif strategy_name == "phase_sweep":
        # Sweep de phase avec amplitude fixe
        ox = np.random.uniform(0.5, 1.0)
        oy = np.random.uniform(0.5, 1.0)
        phases = np.linspace(0, 2*np.pi, n_configs, endpoint=False)
        for phase in phases:
            configs.append(((ox, oy, 0.0), phase))
    
    elif strategy_name == "amplitude_sweep":
        # Sweep d'amplitude avec phase fixe
        phase = np.deg2rad(np.random.uniform(0, 360))
        amplitudes = np.linspace(0.2, 1.5, n_configs)
        for amp in amplitudes:
            ox = amp
            oy = amp * np.random.uniform(0.5, 1.5)
            configs.append(((ox, oy, 0.0), phase))
    
    elif strategy_name == "orthogonal":
        # Configs orthogonales pour couvrir l'espace
        configs = [
            ((1.0, 0.2, 0.0), np.deg2rad(0)),
            ((0.2, 1.0, 0.0), np.deg2rad(90)),
            ((0.8, 0.8, 0.0), np.deg2rad(45)),
            ((0.5, 1.2, 0.0), np.deg2rad(180)),
            ((1.2, 0.5, 0.0), np.deg2rad(270)),
        ]
        configs = configs[:n_configs]
    
    elif strategy_name == "mixed":
        # Mélange de différentes stratégies
        configs.append(((1.0, 0.2, 0.0), np.deg2rad(180)))  # linéaire X
        configs.append(((0.2, 1.0, 0.0), np.deg2rad(180)))  # linéaire Y
        configs.append(((np.random.uniform(0.3, 0.8), 
                        np.random.uniform(0.3, 0.8), 0.0), 
                       np.deg2rad(np.random.uniform(0, 360))))  # random
        configs.append(((0.8, 0.7, 0.0), np.deg2rad(120)))  # promising
        configs.append(((0.2, 0.6, 0.0), np.deg2rad(60)))   # promising
    
    else:  # default = "focused"
        return generate_mw_configs_strategy("focused", n_configs, seed)
    
    return configs


def get_strategy_for_iteration(iteration):
    """Retourne une stratégie différente à chaque itération."""
    strategies = [
        "random",
        "focused",
        "ellipticity_sweep",
        "phase_sweep",
        "amplitude_sweep",
        "orthogonal",
        "mixed",
    ]
    return strategies[iteration % len(strategies)]


# ===================================
# GÉNÉRATION DE DATASET
# ===================================

def compute_spectra_for_B_config(args):
    """Calcule les spectres pour une config B donnée."""
    b_idx, B_field, mw_configs, freq_list, tilt_x_deg, tilt_y_deg = args
    
    n_mw_configs = len(mw_configs)
    freq_points = len(freq_list)
    all_spectra = np.zeros((n_mw_configs, freq_points))
    
    for mw_idx, mw_config in enumerate(mw_configs):
        MW_field, MW_phase = mw_config
        MW_field = list(MW_field)
        
        odmr_spectrum = Ensemble_Spectrum(
            B_field,
            MW_field,
            MW_phase,
            freq_list,
            np.deg2rad(tilt_x_deg),
            np.deg2rad(tilt_y_deg)
        )
        all_spectra[mw_idx, :] = odmr_spectrum
    
    return b_idx, all_spectra, B_field


def generate_dataset(mw_configs, b_configs, output_dir, freq_list, tilt_x, tilt_y):
    """
    Génère un dataset complet de spectres ODMR.
    
    Parameters:
        mw_configs : list
            Liste de configs MW [(MW_field, MW_phase), ...]
        b_configs : ndarray
            Configs de champ magnétique (N, 3)
        output_dir : str
            Dossier de sortie
        freq_list : ndarray
            Liste des fréquences
        tilt_x, tilt_y : float
            Inclinaisons en degrés
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "signals"), exist_ok=True)
    
    n_configs = len(b_configs)
    
    # Préparation des arguments pour multiprocessing
    args_list = [
        (i, b_configs[i].tolist(), mw_configs, freq_list, tilt_x, tilt_y)
        for i in range(n_configs)
    ]
    
    # Calcul parallèle
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(compute_spectra_for_B_config, args_list),
            total=n_configs,
            desc="Generating dataset"
        ))
    
    # Sauvegarde
    for b_idx, spectra, B_field in results:
        signal_path = os.path.join(output_dir, "signals", f"config_{b_idx:05d}.npy")
        np.save(signal_path, spectra)
    
    # Sauvegarde des métadonnées
    np.save(os.path.join(output_dir, "frequencies.npy"), freq_list)
    
    # Sauvegarde MW configs
    mw_configs_array = np.array([
        [mw[0][0], mw[0][1], mw[0][2], mw[1]]
        for mw in mw_configs
    ])
    np.save(os.path.join(output_dir, "mw_configs.npy"), mw_configs_array)
    
    # Sauvegarde configs.csv
    import pandas as pd
    configs_df = pd.DataFrame(b_configs, columns=["Bx", "By", "Bz"])
    configs_df.to_csv(os.path.join(output_dir, "configs.csv"), index=False)
    
    print(f"Dataset generated in {output_dir}")
    print(f"  - {n_configs} B configurations")
    print(f"  - {len(mw_configs)} MW configurations")
    print(f"  - {len(freq_list)} frequency points")


# ===================================
# ENTRAÎNEMENT
# ===================================

class SimpleTrainer:
    """Entraîneur simplifié pour la pipeline."""
    
    def __init__(self, model, device, learning_rate=0.001, scaler_y=None):
        self.model = model.to(device)
        self.device = device
        self.scaler_y = scaler_y
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20
        )
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for signals, B_fields in train_loader:
            signals = signals.to(self.device)
            B_fields = B_fields.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(signals)
            loss = self.criterion(predictions, B_fields)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for signals, B_fields in val_loader:
                signals = signals.to(self.device)
                B_fields = B_fields.to(self.device)
                
                predictions = self.model(signals)
                loss = self.criterion(predictions, B_fields)
                
                total_loss += loss.item()
                n_batches += 1
        
        val_loss = total_loss / n_batches
        self.scheduler.step(val_loss)
        
        return val_loss
    
    def train(self, train_loader, val_loader, n_epochs, patience):
        """Entraîne le modèle avec early stopping."""
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.best_val_loss


def evaluate_model_on_test(model, test_loader, scaler_y, device):
    """
    Évalue le modèle sur le test set.
    
    Returns:
        metrics : dict
            Dictionnaire avec toutes les métriques
    """
    model.eval()
    
    B_preds = []
    B_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            B_preds.append(outputs.cpu().numpy())
            B_labels.append(labels.cpu().numpy())
    
    B_preds = np.concatenate(B_preds, axis=0)
    B_labels = np.concatenate(B_labels, axis=0)
    
    # Inverse transform
    B_preds_inv = scaler_y.inverse_transform(B_preds)
    B_labels_inv = scaler_y.inverse_transform(B_labels)
    
    errors = B_preds_inv - B_labels_inv
    
    # Métriques
    rmse_bx, rmse_by, rmse_bz = np.sqrt(np.mean(errors**2, axis=0)) * 1000  # T -> mT
    avg_rmse = (rmse_bx + rmse_by + rmse_bz) / 3
    
    norm_true = np.linalg.norm(B_labels_inv, axis=1)
    norm_pred = np.linalg.norm(B_preds_inv, axis=1)
    rmse_norm = np.sqrt(np.mean((norm_pred - norm_true)**2)) * 1000
    
    dot_products = np.sum(B_preds_inv * B_labels_inv, axis=1)
    norms_product = np.linalg.norm(B_preds_inv, axis=1) * np.linalg.norm(B_labels_inv, axis=1)
    cos_angles = np.clip(dot_products / norms_product, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    mean_angle_deg = np.degrees(np.mean(angles))
    
    return {
        "rmse_bx": float(rmse_bx),
        "rmse_by": float(rmse_by),
        "rmse_bz": float(rmse_bz),
        "avg_rmse": float(avg_rmse),
        "rmse_norm": float(rmse_norm),
        "mean_angle_deg": float(mean_angle_deg),
    }


# ===================================
# PIPELINE PRINCIPALE
# ===================================

def run_single_iteration(iteration, b_configs, freq_list):
    """
    Exécute une itération complète de la pipeline :
    1. Génère des configs MW
    2. Génère le dataset
    3. Entraîne les modèles
    4. Évalue et retourne le meilleur
    
    Returns:
        result : dict
            Résultats de l'itération
    """
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration + 1}/{N_ITERATIONS}")
    print(f"{'='*60}")
    
    # 1. Générer configs MW selon stratégie
    strategy = get_strategy_for_iteration(iteration)
    print(f"Strategy: {strategy}")
    
    mw_configs = generate_mw_configs_strategy(
        strategy,
        n_configs=N_MW_CONFIGS,
        seed=iteration
    )
    
    print(f"Generated MW configs:")
    for i, (mw_field, phase) in enumerate(mw_configs):
        print(f"  Config {i+1}: Ox={mw_field[0]:.2f}, Oy={mw_field[1]:.2f}, Phase={np.rad2deg(phase):.1f}°")
    
    # 2. Générer dataset
    print("\nGenerating dataset...")
    dataset_dir = f"{TEMP_DATASET_DIR}_iter_{iteration}"
    
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    generate_dataset(
        mw_configs=mw_configs,
        b_configs=b_configs,
        output_dir=dataset_dir,
        freq_list=freq_list,
        tilt_x=TILT_X,
        tilt_y=TILT_Y
    )
    
    # 3. Charger dataset
    print("\nLoading dataset...")
    X, y, frequencies, mw_configs_loaded, scaler_X, scaler_y = load_odmr_dataset(
        dataset_dir=dataset_dir,
        flatten=False,
        normalize=True
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=BATCH_SIZE
    )
    
    # 4. Entraîner et évaluer chaque modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    best_model_result = None
    best_metric = float('inf')
    
    for model_type in MODELS_TO_TEST:
        print(f"\n--- Training {model_type.upper()} ---")
        
        model = get_model(model_type, N_MW_CONFIGS, FREQ_POINTS).to(device)
        n_params = count_parameters(model)
        print(f"Parameters: {n_params:,}")
        
        trainer = SimpleTrainer(model, device, LEARNING_RATE, scaler_y)
        
        print("Training...")
        best_val_loss = trainer.train(train_loader, val_loader, N_EPOCHS, PATIENCE)
        
        print("Evaluating on test set...")
        metrics = evaluate_model_on_test(model, test_loader, scaler_y, device)
        
        print(f"Test RMSE: Bx={metrics['rmse_bx']:.3f} mT, "
              f"By={metrics['rmse_by']:.3f} mT, Bz={metrics['rmse_bz']:.3f} mT")
        print(f"Average RMSE: {metrics['avg_rmse']:.3f} mT")
        print(f"RMSE |B|: {metrics['rmse_norm']:.3f} mT")
        print(f"Angular error: {metrics['mean_angle_deg']:.2f}°")
        
        # Sauvegarder si meilleur
        current_metric = metrics[OPTIMIZATION_METRIC]
        if current_metric < best_metric:
            best_metric = current_metric
            best_model_result = {
                "iteration": iteration,
                "strategy": strategy,
                "model_type": model_type,
                "n_parameters": n_params,
                "mw_configs": mw_configs,
                "metrics": metrics,
                "model_state_dict": model.state_dict(),
                "scaler_y": scaler_y,
                "val_loss": best_val_loss,
            }
    
    # Nettoyer dataset temporaire
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    
    return best_model_result


def save_best_results(result):
    """Sauvegarde les meilleurs résultats."""
    os.makedirs(BEST_RESULTS_DIR, exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = os.path.join(BEST_RESULTS_DIR, "best_model.pth")
    torch.save({
        "model_state_dict": result["model_state_dict"],
        "model_type": result["model_type"],
        "n_parameters": result["n_parameters"],
    }, model_path)
    
    # Sauvegarder les configs MW
    mw_configs_array = np.array([
        [mw[0][0], mw[0][1], mw[0][2], mw[1]]
        for mw in result["mw_configs"]
    ])
    np.save(os.path.join(BEST_RESULTS_DIR, "best_mw_configs.npy"), mw_configs_array)
    
    # Sauvegarder scaler
    import pickle
    with open(os.path.join(BEST_RESULTS_DIR, "scaler_y.pkl"), "wb") as f:
        pickle.dump(result["scaler_y"], f)
    
    # Sauvegarder les résultats en JSON
    results_json = {
        "iteration": result["iteration"],
        "strategy": result["strategy"],
        "model_type": result["model_type"],
        "n_parameters": result["n_parameters"],
        "metrics": result["metrics"],
        "val_loss": result["val_loss"],
        "mw_configs": [
            {
                "ox": float(mw[0][0]),
                "oy": float(mw[0][1]),
                "oz": float(mw[0][2]),
                "phase_rad": float(mw[1]),
                "phase_deg": float(np.rad2deg(mw[1])),
            }
            for mw in result["mw_configs"]
        ],
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(os.path.join(BEST_RESULTS_DIR, "best_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Best results saved to {BEST_RESULTS_DIR}")


def run_optimization_pipeline():
    """Pipeline d'optimisation principale."""
    print("="*60)
    print("OPTIMIZATION PIPELINE")
    print("="*60)
    print(f"Iterations: {N_ITERATIONS}")
    print(f"B configs: {N_B_CONFIGS}")
    print(f"MW configs per dataset: {N_MW_CONFIGS}")
    print(f"Models: {MODELS_TO_TEST}")
    print(f"Optimization metric: {OPTIMIZATION_METRIC}")
    print("="*60)
    
    # Charger ou générer les configs B
    if os.path.exists(B_CONFIGS_FILE):
        print(f"\nLoading B configs from {B_CONFIGS_FILE}")
        b_configs = np.load(B_CONFIGS_FILE)
        if len(b_configs) < N_B_CONFIGS:
            print(f"Warning: only {len(b_configs)} configs available, using all")
        else:
            b_configs = b_configs[:N_B_CONFIGS]
    else:
        print(f"\nGenerating {N_B_CONFIGS} random B configs...")
        b_configs = np.random.uniform(-0.01, 0.01, size=(N_B_CONFIGS, 3))
    
    print(f"Using {len(b_configs)} B configurations")
    
    # Générer liste de fréquences
    freq_list = np.linspace(FREQ_START, FREQ_END, FREQ_POINTS)
    
    # Tracking du meilleur
    global_best_metric = float('inf')
    global_best_result = None
    
    # Historique
    history = []
    
    # Boucle d'optimisation
    for iteration in range(N_ITERATIONS):
        try:
            result = run_single_iteration(iteration, b_configs, freq_list)
            
            if result is None:
                print(f"Iteration {iteration} failed, skipping...")
                continue
            
            current_metric = result["metrics"][OPTIMIZATION_METRIC]
            history.append({
                "iteration": iteration,
                "strategy": result["strategy"],
                "model_type": result["model_type"],
                "metric": current_metric,
            })
            
            # Comparaison avec le meilleur global
            if current_metric < global_best_metric:
                global_best_metric = current_metric
                global_best_result = result
                
                print(f"\n🎉 NEW BEST RESULT! {OPTIMIZATION_METRIC} = {current_metric:.3f}")
                save_best_results(result)
            else:
                print(f"\nNo improvement. Best remains {global_best_metric:.3f}")
        
        except Exception as e:
            print(f"\n❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Résumé final
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    if global_best_result:
        print(f"Best iteration: {global_best_result['iteration']}")
        print(f"Best strategy: {global_best_result['strategy']}")
        print(f"Best model: {global_best_result['model_type']}")
        print(f"Best {OPTIMIZATION_METRIC}: {global_best_metric:.3f}")
        print(f"\nFull metrics:")
        for key, value in global_best_result["metrics"].items():
            print(f"  {key}: {value:.3f}")
        
        print(f"\nBest MW configs:")
        for i, (mw_field, phase) in enumerate(global_best_result["mw_configs"]):
            print(f"  Config {i+1}: Ox={mw_field[0]:.3f}, Oy={mw_field[1]:.3f}, "
                  f"Phase={np.rad2deg(phase):.1f}°")
    
    # Sauvegarder l'historique
    history_path = os.path.join(BEST_RESULTS_DIR, "optimization_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nHistory saved to {history_path}")
    print(f"Best results saved to {BEST_RESULTS_DIR}")


if __name__ == "__main__":
    run_optimization_pipeline()

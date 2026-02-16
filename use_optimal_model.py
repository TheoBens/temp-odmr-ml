"""
Script utilitaire pour charger et utiliser le meilleur modèle trouvé
par la pipeline d'optimisation.
"""

import torch
import numpy as np
import pickle
import json
import os
from models import get_model


class OptimalODMRPredictor:
    """
    Classe pour charger et utiliser le meilleur modèle trouvé.
    """
    
    def __init__(self, results_dir="best_optimization_results"):
        """
        Charge le meilleur modèle et ses configurations.
        
        Parameters:
            results_dir : str
                Dossier contenant les résultats d'optimisation
        """
        self.results_dir = results_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger les résultats
        with open(os.path.join(results_dir, "best_results.json"), "r") as f:
            self.results = json.load(f)
        
        # Charger le scaler
        with open(os.path.join(results_dir, "scaler_y.pkl"), "rb") as f:
            self.scaler_y = pickle.load(f)
        
        # Charger les configs MW
        self.mw_configs = np.load(os.path.join(results_dir, "best_mw_configs.npy"))
        
        # Charger le modèle
        model_type = self.results['model_type']
        n_mw_configs = len(self.mw_configs)
        n_freq_points = 250  # À ajuster si différent
        
        self.model = get_model(model_type, n_mw_configs, n_freq_points).to(self.device)
        
        checkpoint = torch.load(
            os.path.join(results_dir, "best_model.pth"),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        print(f"✓ Loaded optimal model: {model_type}")
        print(f"✓ Iteration: {self.results['iteration']}")
        print(f"✓ Strategy: {self.results['strategy']}")
        print(f"✓ Performance: {self.results['metrics']['avg_rmse']:.3f} mT (avg RMSE)")
    
    def predict(self, spectra, normalize=True):
        """
        Prédit le champ magnétique à partir de spectres ODMR.
        
        Parameters:
            spectra : ndarray, shape (n_samples, n_mw_configs, n_freq_points)
                Spectres ODMR normalisés ou non
            normalize : bool
                Si True, normalise les spectres en entrée
        
        Returns:
            B_pred : ndarray, shape (n_samples, 3)
                Prédictions de champ magnétique [Bx, By, Bz] en Tesla
        """
        # Convertir en tensor
        if isinstance(spectra, np.ndarray):
            spectra_tensor = torch.from_numpy(spectra).float()
        else:
            spectra_tensor = spectra.float()
        
        # Normaliser si nécessaire
        if normalize:
            # Normalisation simple par spectre
            mean = spectra_tensor.mean(dim=-1, keepdim=True)
            std = spectra_tensor.std(dim=-1, keepdim=True) + 1e-8
            spectra_tensor = (spectra_tensor - mean) / std
        
        # Prédire
        spectra_tensor = spectra_tensor.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(spectra_tensor)
        
        predictions_np = predictions.cpu().numpy()
        
        # Dénormaliser
        B_pred = self.scaler_y.inverse_transform(predictions_np)
        
        return B_pred
    
    def predict_single(self, spectrum, normalize=True):
        """
        Prédit le champ magnétique pour un seul spectre.
        
        Parameters:
            spectrum : ndarray, shape (n_mw_configs, n_freq_points)
                Un seul spectre ODMR
            normalize : bool
                Si True, normalise le spectre
        
        Returns:
            B_pred : ndarray, shape (3,)
                Prédiction [Bx, By, Bz] en Tesla
        """
        spectrum_batch = np.expand_dims(spectrum, axis=0)
        B_pred_batch = self.predict(spectrum_batch, normalize=normalize)
        return B_pred_batch[0]
    
    def get_optimal_mw_configs(self):
        """
        Retourne les configurations MW optimales.
        
        Returns:
            mw_configs : list of dict
                Liste des configs MW avec Ox, Oy, Oz, phase
        """
        configs = []
        for i, config in enumerate(self.results['mw_configs']):
            configs.append({
                'id': i + 1,
                'ox': config['ox'],
                'oy': config['oy'],
                'oz': config['oz'],
                'phase_rad': config['phase_rad'],
                'phase_deg': config['phase_deg'],
            })
        return configs
    
    def get_metrics(self):
        """Retourne les métriques de performance."""
        return self.results['metrics']
    
    def summary(self):
        """Affiche un résumé du modèle optimal."""
        print("\n" + "="*70)
        print("OPTIMAL MODEL SUMMARY")
        print("="*70)
        
        print(f"\nModel: {self.results['model_type']}")
        print(f"Parameters: {self.results['n_parameters']:,}")
        print(f"Optimization iteration: {self.results['iteration']}")
        print(f"Strategy used: {self.results['strategy']}")
        
        print(f"\n{'PERFORMANCE':^70}")
        print("-"*70)
        metrics = self.results['metrics']
        print(f"RMSE |B|    : {metrics['rmse_norm']:.3f} mT")
        print(f"RMSE Bx     : {metrics['rmse_bx']:.3f} mT")
        print(f"RMSE By     : {metrics['rmse_by']:.3f} mT")
        print(f"RMSE Bz     : {metrics['rmse_bz']:.3f} mT")
        print(f"Average RMSE: {metrics['avg_rmse']:.3f} mT")
        print(f"Angular err : {metrics['mean_angle_deg']:.2f}°")
        
        print(f"\n{'OPTIMAL MW CONFIGURATIONS':^70}")
        print("-"*70)
        for config in self.get_optimal_mw_configs():
            print(f"Config {config['id']}: "
                  f"Ox={config['ox']:.3f}, Oy={config['oy']:.3f}, "
                  f"Phase={config['phase_deg']:.1f}°")
        
        print("="*70 + "\n")


def demo_usage():
    """Démonstration d'utilisation du modèle optimal."""
    
    # Charger le modèle
    predictor = OptimalODMRPredictor()
    
    # Afficher le résumé
    predictor.summary()
    
    # Afficher les configs MW optimales
    print("\n" + "="*70)
    print("OPTIMAL MW CONFIGURATIONS (for dataset generation)")
    print("="*70)
    mw_configs = predictor.get_optimal_mw_configs()
    print("\nCopy this to your dataset generation script:\n")
    print("MW_CONFIGS = [")
    for config in mw_configs:
        print(f"    (({config['ox']:.2f}, {config['oy']:.2f}, {config['oz']:.2f}), "
              f"np.deg2rad({config['phase_deg']:.1f})),")
    print("]\n")
    
    # Exemple de prédiction (simulé)
    print("="*70)
    print("EXAMPLE PREDICTION")
    print("="*70)
    
    # Créer des spectres fictifs
    n_mw_configs = len(mw_configs)
    n_freq_points = 250
    fake_spectra = np.random.randn(5, n_mw_configs, n_freq_points)
    
    # Prédire
    B_predictions = predictor.predict(fake_spectra, normalize=True)
    
    print("\nPredictions (using random spectra for demo):")
    for i, B in enumerate(B_predictions):
        norm = np.linalg.norm(B)
        print(f"Sample {i+1}: Bx={B[0]*1000:.2f} mT, By={B[1]*1000:.2f} mT, "
              f"Bz={B[2]*1000:.2f} mT, |B|={norm*1000:.2f} mT")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Vérifier si les résultats existent
    if not os.path.exists("best_optimization_results/best_results.json"):
        print("❌ No optimization results found!")
        print("Please run 'python optimize_pipeline.py' first.")
    else:
        demo_usage()

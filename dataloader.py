import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm


class ODMRDataset(Dataset):
    """
    PyTorch Dataset pour les spectres ODMR.
    
    Input: Spectres ODMR (N_MW_CONFIGS x FREQ_POINTS)
    Output: Champ magnétique [Bx, By, Bz]
    """
    
    def __init__(self, signals, magnetic_fields, transform=None):
        """
        Parameters:
        -----------
        signals : array
            Spectres ODMR (shape: N_samples x N_MW_CONFIGS x FREQ_POINTS)
        magnetic_fields : array
            Champs magnétiques (shape: N_samples x 3)
        transform : callable, optional
            Transformation à appliquer aux données
        """
        self.signals = torch.FloatTensor(signals)
        self.magnetic_fields = torch.FloatTensor(magnetic_fields)
        self.transform = transform
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        B_field = self.magnetic_fields[idx]
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, B_field


def load_odmr_dataset(dataset_dir, flatten=False, normalize=True, max_mw_configs=None):
    """
    Charge le dataset ODMR complet.
    
    Parameters:
    -----------
    dataset_dir : str
        Dossier contenant le dataset
    flatten : bool
        Si True, flatten les spectres (N_MW*FREQ_POINTS,)
        Si False, garde la structure 2D (N_MW, FREQ_POINTS)
    normalize : bool
        Normaliser les spectres et champs magnétiques
    max_mw_configs : int, optional
        Nombre maximum de configurations MW à utiliser (garde les N premières)
        Si None, utilise toutes les configurations
    
    Returns:
    --------
    X : array
        Spectres ODMR
    y : array
        Champs magnétiques [Bx, By, Bz]
    frequencies : array
        Fréquences
    mw_configs : array
        Configurations MW
    """
    # Charger les configurations
    configs_path = os.path.join(dataset_dir, "configs.csv")
    configs_df = pd.read_csv(configs_path)
    
    # Charger les fréquences et configs MW
    frequencies = np.load(os.path.join(dataset_dir, "frequencies.npy"))
    mw_configs = np.load(os.path.join(dataset_dir, "mw_configs.npy"))
    
    # Charger tous les spectres
    signals_dir = os.path.join(dataset_dir, "signals")
    signal_files = sorted([f for f in os.listdir(signals_dir) if f.endswith('.npy')])
    
    n_samples = len(signal_files)
    
    # Charger le premier fichier pour déterminer les dimensions
    first_signal = np.load(os.path.join(signals_dir, signal_files[0]))
    n_mw_configs_full, n_freq_points = first_signal.shape
    
    # Limiter le nombre de configs MW si spécifié
    if max_mw_configs is not None:
        n_mw_configs = min(max_mw_configs, n_mw_configs_full)
        print(f"Utilisation de {n_mw_configs}/{n_mw_configs_full} configurations MW")
    else:
        n_mw_configs = n_mw_configs_full
    
    # Initialiser les tableaux
    if flatten:
        X = np.zeros((n_samples, n_mw_configs * n_freq_points))
    else:
        X = np.zeros((n_samples, n_mw_configs, n_freq_points))
    
    y = np.zeros((n_samples, 3))
    
    # Charger tous les spectres et champs magnétiques
    print("Chargement du dataset", dataset_dir)
    for i, signal_file in enumerate(tqdm(signal_files, desc="Chargement")):
        # Charger le spectre
        signal = np.load(os.path.join(signals_dir, signal_file))
        
        # Ne garder que les N premières configs MW si spécifié
        if max_mw_configs is not None:
            signal = signal[:max_mw_configs, :]
        
        if flatten:
            X[i] = signal.flatten()
        else:
            X[i] = signal
        
        # Récupérer le champ magnétique correspondant
        config = configs_df.iloc[i]
        y[i] = [config['Bx'], config['By'], config['Bz']]
    
    # Normalisation
    if normalize:
        print("\nNormalisation des données...")
        # Normaliser les spectres
        X_shape = X.shape
        X_flat = X.reshape(n_samples, -1)
        scaler_X = StandardScaler()
        X_flat = scaler_X.fit_transform(X_flat)
        X = X_flat.reshape(X_shape)
        
        # Normaliser les champs magnétiques
        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)
        
        # Ne retourner que les configs MW utilisées
        if max_mw_configs is not None:
            mw_configs = mw_configs[:max_mw_configs]
        
        return X, y, frequencies, mw_configs, scaler_X, scaler_y
    
    # Ne retourner que les configs MW utilisées
    if max_mw_configs is not None:
        mw_configs = mw_configs[:max_mw_configs]
    
    return X, y, frequencies, mw_configs, None, None


def create_train_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Sépare les données en ensembles d'entraînement, validation et test.
    
    Parameters:
    -----------
    X : array
        Spectres ODMR
    y : array
        Champs magnétiques
    test_size : float
        Proportion de l'ensemble de test
    val_size : float
        Proportion de l'ensemble de validation (sur les données d'entraînement)
    random_state : int
        Graine aléatoire
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Split train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Split train / val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"\nRépartition des données:")
    print(f"  Train: {len(X_train)} échantillons ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} échantillons ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} échantillons ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """
    Crée les DataLoaders PyTorch.
    
    Parameters:
    -----------
    X_train, y_train, X_val, y_val, X_test, y_test : arrays
        Données d'entraînement, validation et test
    batch_size : int
        Taille des batches
    
    Returns:
    --------
    train_loader, val_loader, test_loader
    """
    train_dataset = ODMRDataset(X_train, y_train)
    val_dataset = ODMRDataset(X_val, y_val)
    test_dataset = ODMRDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    dataset_dir = "odmr_synthetic_dataset"

    # Charger le dataset
    X, y, frequencies, mw_configs, scaler_X, scaler_y = load_odmr_dataset(
        dataset_dir, 
        flatten=False,
        normalize=True
    )
    
    print(f"\n{'='*70}")
    print("Informations sur le dataset:")
    print(f"{'='*70}")
    print(f"Shape des spectres (X): {X.shape}")
    print(f"Shape des champs magnétiques (y): {y.shape}")
    print(f"Nombre de fréquences: {len(frequencies)}")
    print(f"Nombre de configs MW: {len(mw_configs)}")
    print(f"\nPlage des valeurs normalisées:")
    print(f"  X: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  y: [{y.min():.3f}, {y.max():.3f}]")
    
    # Créer les splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
    
    # Créer les dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=16
    )
    
    print(f"\n{'='*70}")
    print("DataLoaders créés avec succès!")
    print(f"{'='*70}")
    print(f"Taille des batches: 16")
    print(f"Nombre de batches (train): {len(train_loader)}")
    print(f"Nombre de batches (val): {len(val_loader)}")
    print(f"Nombre de batches (test): {len(test_loader)}")

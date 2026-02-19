import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from dataloader import load_odmr_dataset, create_train_test_split, create_dataloaders
from models import get_model, count_parameters


class ODMRTrainer:
    """Classe pour gérer l'entraînement des modèles."""
    
    def __init__(self, model, device='cuda', learning_rate=0.001, save_dir='models', scaler_y=None):
        """
        Parameters:
        -----------
        model : nn.Module
            Modèle PyTorch
        device : str
            'cuda' ou 'cpu'
        learning_rate : float
            Taux d'apprentissage
        save_dir : str
            Dossier pour sauvegarder les modèles
        scaler_y : StandardScaler
            Scaler pour dénormaliser les prédictions
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.scaler_y = scaler_y
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimiseur et loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20
        )
        self.criterion = nn.MSELoss()
        
        # Historique
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rate': []
        }
        
    def train_epoch(self, train_loader):
        """Entraîne le modèle pendant une époque."""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_squared_error_real = np.zeros(3)  # Erreur sur données réelles (Tesla)
        total_samples = 0
        n_batches = 0
        
        for signals, B_fields in train_loader:
            signals = signals.to(self.device)
            B_fields = B_fields.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(signals)
            
            # Loss
            loss = self.criterion(predictions, B_fields)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - B_fields)).item()
            
            # RMSE sur données réelles (dénormalisées)
            if self.scaler_y is not None:
                B_pred_real = self.scaler_y.inverse_transform(predictions.detach().cpu().numpy())
                B_true_real = self.scaler_y.inverse_transform(B_fields.detach().cpu().numpy())
                error_real = B_pred_real - B_true_real
                total_squared_error_real += np.sum(error_real**2, axis=0)
            
            total_samples += B_fields.size(0)
            n_batches += 1
        
        # Mean
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        
        # RMSE en Tesla (converti en mT pour affichage)
        if self.scaler_y is not None:
            rmse_components = np.sqrt(total_squared_error_real / total_samples)
            avg_rmse_bx, avg_rmse_by, avg_rmse_bz = rmse_components * 1000  # Tesla -> mT
            avg_rmse = np.sqrt(np.sum(total_squared_error_real) / total_samples * 3) * 1000
        else:
            avg_rmse_bx = avg_rmse_by = avg_rmse_bz = avg_rmse = 0.0

        return avg_loss, avg_mae, avg_rmse, avg_rmse_bx, avg_rmse_by, avg_rmse_bz
    
    def validate_epoch(self, val_loader):
        """Valide le modèle."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_squared_error_real = np.zeros(3)  # Erreur sur données réelles (Tesla)
        total_samples = 0
        n_batches = 0
        
        with torch.no_grad():
            for signals, B_fields in val_loader:
                signals = signals.to(self.device)
                B_fields = B_fields.to(self.device)
                
                # Forward pass
                predictions = self.model(signals)
                
                # Loss
                loss = self.criterion(predictions, B_fields)
                
                # Metrics
                total_loss += loss.item()
                total_mae += torch.mean(torch.abs(predictions - B_fields)).item()
                
                # RMSE sur données réelles (dénormalisées)
                if self.scaler_y is not None:
                    B_pred_real = self.scaler_y.inverse_transform(predictions.cpu().numpy())
                    B_true_real = self.scaler_y.inverse_transform(B_fields.cpu().numpy())
                    error_real = B_pred_real - B_true_real
                    total_squared_error_real += np.sum(error_real**2, axis=0)
                
                total_samples += B_fields.size(0)
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_mae = total_mae / n_batches
        
        # RMSE en Tesla (converti en mT pour affichage)
        if self.scaler_y is not None:
            rmse_components = np.sqrt(total_squared_error_real / total_samples)
            avg_rmse_bx, avg_rmse_by, avg_rmse_bz = rmse_components * 1000  # Tesla -> mT
            avg_rmse = np.sqrt(np.sum(total_squared_error_real) / total_samples * 3) * 1000
        else:
            avg_rmse_bx = avg_rmse_by = avg_rmse_bz = avg_rmse = 0.0
        
        return avg_loss, avg_mae, avg_rmse, avg_rmse_bx, avg_rmse_by, avg_rmse_bz
    
    def train(self, train_loader, val_loader, n_epochs=200, early_stopping_patience=20):
        """
        Entraîne le modèle.
        
        Parameters:
        -----------
        train_loader : DataLoader
            Données d'entraînement
        val_loader : DataLoader
            Données de validation
        n_epochs : int
            Nombre d'époques
        early_stopping_patience : int
            Patience pour l'early stopping
        """
        print("="*70)
        print("ENTRAÎNEMENT DU MODÈLE")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Nombre de paramètres: {count_parameters(self.model):,}")
        print(f"Époques: {n_epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("="*70 + "\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Entraînement
            train_loss, train_mae, train_rmse, train_rmse_bx, train_rmse_by, train_rmse_bz = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_mae, val_rmse, val_rmse_bx, val_rmse_by, val_rmse_bz = self.validate_epoch(val_loader)
            
            # Scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Historique
            self.history.setdefault('train_loss', []).append(train_loss)
            self.history.setdefault('val_loss', []).append(val_loss)
            self.history.setdefault('train_mae', []).append(train_mae)
            self.history.setdefault('val_mae', []).append(val_mae)
            self.history.setdefault('learning_rate', []).append(current_lr)
            self.history.setdefault('train_rmse', []).append(train_rmse)
            self.history.setdefault('val_rmse', []).append(val_rmse)
            self.history.setdefault('train_rmse_bx', []).append(train_rmse_bx)
            self.history.setdefault('train_rmse_by', []).append(train_rmse_by)
            self.history.setdefault('train_rmse_bz', []).append(train_rmse_bz)
            self.history.setdefault('val_rmse_bx', []).append(val_rmse_bx)
            self.history.setdefault('val_rmse_by', []).append(val_rmse_by)
            self.history.setdefault('val_rmse_bz', []).append(val_rmse_bz)
            # Affichage
            # print(f"Epoch {epoch+1}/{n_epochs} - "
            #       f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f} | "
            #       f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f} | "
            #       f"Train RMSE: {train_rmse:.6f} mT, Val RMSE: {val_rmse:.6f} mT | "
            #         f"Train RMSE (Bx): {train_rmse_bx:.6f} mT, Val RMSE (Bx): {val_rmse_bx:.6f} mT | "
            #         f"Train RMSE (By): {train_rmse_by:.6f} mT, Val RMSE (By): {val_rmse_by:.6f} mT | "
            #         f"Train RMSE (Bz): {train_rmse_bz:.6f} mT, Val RMSE (Bz): {val_rmse_bz:.6f} mT | "
            #       f"LR: {current_lr:.6f}")
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Val RMSE (Bx) : {val_rmse_bx:.6f} mT, "
                  f"Val RMSE (By): {val_rmse_by:.6f} mT, "
                  f"Val RMSE (Bz): {val_rmse_bz:.6f} mT ")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, val_loss)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping à l'époque {epoch+1}")
                break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best validation loss: {best_val_loss:.6f}")
        
    def save_checkpoint(self, filename, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch'], checkpoint['val_loss']
    
    def plot_history(self, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(15, 12))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('MSE Loss', fontsize=12)
        axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE
        axes[0, 1].plot(self.history['train_mae'], label='Train MAE', linewidth=2)
        axes[0, 1].plot(self.history['val_mae'], label='Val MAE', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('MAE', fontsize=12)
        axes[0, 1].set_title('MAE Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # RMSE
        axes[0, 2].plot(self.history['train_rmse'], label='Train RMSE', linewidth=2)
        axes[0, 2].plot(self.history['val_rmse'], label='Val RMSE', linewidth=2)
        axes[0, 2].set_xlabel('Epoch', fontsize=12)
        axes[0, 2].set_ylabel('RMSE (mT)', fontsize=12)
        axes[0, 2].set_title('Global RMSE Curve', fontsize=14, fontweight='bold')
        axes[0, 2].legend(fontsize=10)
        axes[0, 2].grid(True, alpha=0.3)

        # RMSE Bx
        axes[1, 0].plot(self.history['train_rmse_bx'], label='Train RMSE Bx', linewidth=2)
        axes[1, 0].plot(self.history['val_rmse_bx'], label='Val RMSE Bx', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('RMSE Bx (mT)', fontsize=12)
        axes[1, 0].set_title('RMSE (Bx)', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

        # RMSE By
        axes[1, 1].plot(self.history['train_rmse_by'], label='Train RMSE By', linewidth=2)
        axes[1, 1].plot(self.history['val_rmse_by'], label='Val RMSE By', linewidth=2)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('RMSE By (mT)', fontsize=12)
        axes[1, 1].set_title('RMSE (By)', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

        # RMSE Bz
        axes[1, 2].plot(self.history['train_rmse_bz'], label='Train RMSE Bz', linewidth=2)
        axes[1, 2].plot(self.history['val_rmse_bz'], label='Val RMSE Bz', linewidth=2)
        axes[1, 2].set_xlabel('Epoch', fontsize=12)
        axes[1, 2].set_ylabel('RMSE Bz (mT)', fontsize=12)
        axes[1, 2].set_title('RMSE (Bz)', fontsize=14, fontweight='bold')
        axes[1, 2].legend(fontsize=10)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def train_model(dataset_dir='odmr_synthetic_dataset', 
                model_type='densenet',
                batch_size=16,
                n_epochs=200,
                learning_rate=0.001,
                save_dir='models'):
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    X, y, frequencies, mw_configs, scaler_X, scaler_y = load_odmr_dataset(
        dataset_dir, 
        flatten=False,
        normalize=True
    )
    
    # Splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)
    
    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    
    # Create model
    n_mw_configs = X.shape[1]
    n_freq_points = X.shape[2]
    model = get_model(model_type, n_mw_configs, n_freq_points)
    
    # Train
    trainer = ODMRTrainer(model, device=device, learning_rate=learning_rate, save_dir=save_dir, scaler_y=scaler_y)
    trainer.train(train_loader, val_loader, n_epochs=n_epochs)
    
    # Save scalers
    import joblib
    joblib.dump(scaler_X, os.path.join(save_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))
    
    # Save configuration
    config = {
        'model_type': model_type,
        'n_mw_configs': n_mw_configs,
        'n_freq_points': n_freq_points,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Plot
    trainer.plot_history(save_path=os.path.join(save_dir, 'training_history.png'))
    
    # Extract histories
    train_history = {
        'loss': trainer.history['train_loss'],
        'mae': trainer.history['train_mae']
    }
    val_history = {
        'loss': trainer.history['val_loss'],
        'mae': trainer.history['val_mae']
    }
    
    return train_history, val_history, trainer.model, test_loader, scaler_y


if __name__ == "__main__":

    train_history, val_history, model, test_loader, scaler_y = train_model(
        dataset_dir='datasets/odmr_synthetic_dataset_2',
        model_type='cnn1d',
        batch_size=16,
        n_epochs=200,
        learning_rate=0.0007,
        save_dir='models_trained/models_cnn1d_2'
    )
    
    print(f"✓ Model saved in: models_trained/models_cnn1d_2/")
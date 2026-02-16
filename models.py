import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Modèles de réseaux de neurones pour la prédiction de champ magnétique
à partir de spectres ODMR.
"""


class ODMR_Simple(nn.Module):
    """
    Modèle SIMPLE avec très peu de paramètres pour éviter le surapprentissage.
    Adapté pour petit dataset (< 1000 échantillons).
    
    Architecture minimaliste:
    - Conv1D légère sur chaque MW config
    - Pooling agressif
    - Très peu de couches denses
    """
    
    def __init__(self, n_mw_configs=4, n_freq_points=201):
        super(ODMR_Simple, self).__init__()
        
        self.n_mw_configs = n_mw_configs
        
        # Une seule couche Conv1D par MW config
        self.conv1 = nn.Conv1d(1, 16, kernel_size=11, padding=5)  # Kernel large pour capturer pics
        self.pool1 = nn.MaxPool1d(4)  # Pooling agressif
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, padding=3)
        self.pool2 = nn.MaxPool1d(4)  # Pooling agressif
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Features après conv: 32 par MW config
        conv_output_dim = 32 * n_mw_configs
        
        # Couches denses MINIMALES
        self.fc1 = nn.Linear(conv_output_dim, 64)
        self.dropout1 = nn.Dropout(0.5)  # Dropout fort
        
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        
        # Sortie directe
        self.output_layer = nn.Linear(32, 3)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        features = []
        for i in range(self.n_mw_configs):
            mw_signal = x[:, i:i+1, :]
            
            # Conv + pool
            out = F.relu(self.conv1(mw_signal))
            out = self.pool1(out)
            
            out = F.relu(self.conv2(out))
            out = self.pool2(out)
            
            # Global pooling
            out = self.global_pool(out)
            out = out.view(batch_size, -1)
            
            features.append(out)
        
        # Concat
        x = torch.cat(features, dim=1)
        
        # Dense minimal
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.output_layer(x)
        
        return x


class ODMR_LinearRegression(nn.Module):
    """
    Régression linéaire pure - baseline absolu.
    Utile pour comprendre si le problème est solvable.
    """
    
    def __init__(self, n_mw_configs=4, n_freq_points=201):
        super(ODMR_LinearRegression, self).__init__()
        
        input_dim = n_mw_configs * n_freq_points
        self.linear = nn.Linear(input_dim, 3)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.linear(x)


class ODMR_DenseNet(nn.Module):
    """
    Réseau de neurones dense (Fully Connected) pour la prédiction de champ magnétique.
    
    Architecture:
    - Flatten des spectres
    - Plusieurs couches denses avec BatchNorm et Dropout
    - Régression vers [Bx, By, Bz]
    """
    
    def __init__(self, n_mw_configs=4, n_freq_points=201, hidden_dims=[512, 256, 128, 64]):
        """
        Parameters:
        -----------
        n_mw_configs : int
            Nombre de configurations MW
        n_freq_points : int
            Nombre de points de fréquence
        hidden_dims : list
            Dimensions des couches cachées
        """
        super(ODMR_DenseNet, self).__init__()
        
        input_dim = n_mw_configs * n_freq_points
        
        # Couches denses
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Couche de sortie (régression vers Bx, By, Bz)
        self.output_layer = nn.Linear(prev_dim, 3)
        
    def forward(self, x):
        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Extraction de features
        x = self.feature_extractor(x)
        
        # Prédiction
        x = self.output_layer(x)
        
        return x


class ODMR_CNN1D(nn.Module):
    """
    Réseau de neurones convolutionnel 1D pour la prédiction de champ magnétique.
    
    Architecture:
    - Conv1D sur chaque configuration MW séparément
    - Global pooling et concatenation
    - Couches denses pour la prédiction finale
    """
    
    def __init__(self, n_mw_configs=4, n_freq_points=201):
        """
        Parameters:
        -----------
        n_mw_configs : int
            Nombre de configurations MW
        n_freq_points : int
            Nombre de points de fréquence
        """
        super(ODMR_CNN1D, self).__init__()
        
        self.n_mw_configs = n_mw_configs
        
        # Couches convolutionnelles (partagées entre les configs MW)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dimensions après les conv + pooling
        conv_output_dim = 128 * n_mw_configs
        
        # Couches denses
        self.fc1 = nn.Linear(conv_output_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        
        # Couche de sortie
        self.output_layer = nn.Linear(64, 3)
        
    def forward(self, x):
        # x shape: (batch, n_mw_configs, n_freq_points)
        batch_size = x.size(0)
        
        # Traiter chaque config MW séparément avec les mêmes poids
        features = []
        for i in range(self.n_mw_configs):
            # Extraire une configuration MW
            mw_signal = x[:, i:i+1, :]  # (batch, 1, n_freq_points)
            
            # Convolutions
            out = F.relu(self.bn1(self.conv1(mw_signal)))
            out = self.pool1(out)
            
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.pool2(out)
            
            out = F.relu(self.bn3(self.conv3(out)))
            
            # Global pooling
            out = self.global_pool(out)  # (batch, 128, 1)
            out = out.view(batch_size, -1)  # (batch, 128)
            
            features.append(out)
        
        # Concaténer les features de toutes les configs MW
        x = torch.cat(features, dim=1)  # (batch, 128 * n_mw_configs)
        
        # Couches denses
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        
        # Prédiction
        x = self.output_layer(x)
        
        return x


class ODMR_Hybrid(nn.Module):
    """
    Modèle hybride combinant CNN2D et Dense layers.
    
    Traite les spectres comme une "image" (N_MW_CONFIGS x FREQ_POINTS).
    """
    
    def __init__(self, n_mw_configs=4, n_freq_points=201):
        super(ODMR_Hybrid, self).__init__()
        
        # Conv2D layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((1, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculer la dimension après convolutions
        # n_mw_configs reste inchangé (padding=(1,2))
        # n_freq_points / 4 après 2 poolings de (1,2)
        conv_h = n_mw_configs - 1  # après conv3 avec kernel (2,3) et padding (0,1)
        conv_w = n_freq_points // 4
        conv_output_dim = 128 * conv_h * conv_w
        
        # Dense layers
        self.fc1 = nn.Linear(conv_output_dim, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        
        self.output_layer = nn.Linear(64, 3)
        
    def forward(self, x):
        # x shape: (batch, n_mw_configs, n_freq_points)
        # Ajouter dimension de canal pour Conv2D
        x = x.unsqueeze(1)  # (batch, 1, n_mw_configs, n_freq_points)
        
        # Convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn_fc3(self.fc3(x)))
        
        # Prédiction
        x = self.output_layer(x)
        
        return x
    
    
class MWConfig_CNN(nn.Module):
    """1D Conv + MW configs aggregation"""
    requires_multi_config = True  # Expects input shape (batch, 10, 201)
    
    def __init__(self, n_channels=10, n_freq=201, output_dim=3):
        super().__init__()
        # Treat each MW config separately
        self.conv_mw = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool freq dim
        
        # Combine MW configs
        self.fc = nn.Sequential(
            nn.Linear(16 * n_channels, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        # x: (batch, 10, 201)
        batch, n_mw, n_freq = x.shape
        x = x.unsqueeze(2)  # (batch, 10, 1, freq)
        x = x.view(batch * n_mw, 1, n_freq)
        
        h = self.conv_mw(x)  # (batch*n_mw, 16, freq)
        h = self.pool(h).squeeze(-1)  # (batch*n_mw, 16)
        h = h.view(batch, n_mw * 16)  # combine MW configs
        
        out = self.fc(h)
        return out



def get_model(model_type='densenet', n_mw_configs=4, n_freq_points=201):
    """
    Factory function pour créer un modèle.
    
    Parameters:
    -----------
    model_type : str
        Type de modèle ('simple', 'linear', 'densenet', 'cnn1d', 'hybrid')
    n_mw_configs : int
        Nombre de configurations MW
    n_freq_points : int
        Nombre de points de fréquence
    
    Returns:
    --------
    model : nn.Module
        Modèle PyTorch
    """
    if model_type == 'simple':
        return ODMR_Simple(n_mw_configs, n_freq_points)
    elif model_type == 'linear':
        return ODMR_LinearRegression(n_mw_configs, n_freq_points)
    elif model_type == 'densenet':
        return ODMR_DenseNet(n_mw_configs, n_freq_points)
    elif model_type == 'cnn1d':
        return ODMR_CNN1D(n_mw_configs, n_freq_points)
    elif model_type == 'hybrid':
        return ODMR_Hybrid(n_mw_configs, n_freq_points)
    elif model_type == 'mw_cnn':
        return MWConfig_CNN(n_channels=n_mw_configs, n_freq=n_freq_points, output_dim=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Compte le nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test des modèles
    print("="*70)
    print("TEST DES MODÈLES")
    print("="*70)
    
    batch_size = 8
    n_mw_configs = 4
    n_freq_points = 201
    
    # Créer un batch de test
    x = torch.randn(batch_size, n_mw_configs, n_freq_points)
    
    models = {
        'DenseNet': get_model('densenet', n_mw_configs, n_freq_points),
        'CNN1D': get_model('cnn1d', n_mw_configs, n_freq_points),
        'Hybrid': get_model('hybrid', n_mw_configs, n_freq_points)
    }
    
    for name, model in models.items():
        print(f"\n{'-'*70}")
        print(f"Modèle: {name}")
        print(f"{'-'*70}")
        
        # Forward pass
        output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Nombre de paramètres: {count_parameters(model):,}")
        
        # Vérifier que la sortie a la bonne forme
        assert output.shape == (batch_size, 3), f"Mauvaise shape de sortie: {output.shape}"
        
    print(f"\n{'='*70}")
    print("Tous les tests ont réussi!")
    print("="*70)

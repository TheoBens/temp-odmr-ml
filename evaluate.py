import torch
import numpy as np
import os
from dataloader import load_odmr_dataset, create_train_test_split, create_dataloaders
from models import get_model


def evaluate_model(dataset_dir="odmr_synthetic_dataset", model_dir="models_hybrid",
                   model_filename="best_model.pth", batch_size=16):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    X, y, frequencies, mw_configs, scaler_X, scaler_y = load_odmr_dataset(dataset_dir=dataset_dir, flatten=False, normalize=True)

    # Create train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_test_split(X, y)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size)

    # Load model
    config_path = os.path.join(model_dir, "config.json")
    checkpoint_path = os.path.join(model_dir, model_filename)

    import json
    with open(config_path, "r") as f:
        config = json.load(f)

    model = get_model(
        config["model_type"],
        config["n_mw_configs"],
        config["n_freq_points"]
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Evaluate on test set
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

    # Inverse transform predictions and labels
    B_preds_inv = scaler_y.inverse_transform(B_preds)
    B_labels_inv = scaler_y.inverse_transform(B_labels)

    errors = B_preds_inv - B_labels_inv

    # =========================
    # Metrics
    # =========================

    # RMSE per component (en Tesla, converti en mT)
    rmse_bx, rmse_by, rmse_bz = np.sqrt(np.mean(errors**2, axis=0)) * 1000  # T -> mT

    # RMSE norm |B| (en Tesla, converti en mT)
    norm_true = np.linalg.norm(B_labels_inv, axis=1)
    norm_pred = np.linalg.norm(B_preds_inv, axis=1)
    rmse_norm = np.sqrt(np.mean((norm_pred - norm_true)**2)) * 1000  # T -> mT

    # Relative RMSE (% of std or mean)
    rel_rmse_bx = rmse_bx / (np.std(B_labels_inv[:, 0]) * 1000) * 100
    rel_rmse_by = rmse_by / (np.std(B_labels_inv[:, 1]) * 1000) * 100
    rel_rmse_bz = rmse_bz / (np.std(B_labels_inv[:, 2]) * 1000) * 100
    rel_rmse_norm = rmse_norm / (np.mean(norm_true) * 1000) * 100

    # Mean angular error (degrees)
    dot_products = np.sum(B_preds_inv * B_labels_inv, axis=1)
    norms_product = np.linalg.norm(B_preds_inv, axis=1) * np.linalg.norm(B_labels_inv, axis=1)
    cos_angles = np.clip(dot_products / norms_product, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    mean_angle_deg = np.degrees(np.mean(angles))

    # Print results
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
    print(f"RMSE |B|           : {rmse_norm:.3f} mT  |  Relative: {rel_rmse_norm:.2f} %")
    print(f"RMSE Bx            : {rmse_bx:.3f} mT  |  Relative: {rel_rmse_bx:.2f} %")
    print(f"RMSE By            : {rmse_by:.3f} mT  |  Relative: {rel_rmse_by:.2f} %")
    print(f"RMSE Bz            : {rmse_bz:.3f} mT  |  Relative: {rel_rmse_bz:.2f} %")
    print(f"Mean angular error : {mean_angle_deg:.2f}°")
    print("="*50)

    return {
        "rmse_norm": rmse_norm,
        "rmse_bx": rmse_bx,
        "rmse_by": rmse_by,
        "rmse_bz": rmse_bz,
        "rel_rmse_norm": rel_rmse_norm,
        "rel_rmse_bx": rel_rmse_bx,
        "rel_rmse_by": rel_rmse_by,
        "rel_rmse_bz": rel_rmse_bz,
        "mean_angle_deg": mean_angle_deg
    }


if __name__ == "__main__":
    evaluate_model('odmr_synthetic_dataset_V2', 'models_linear_V2', 'best_model.pth', batch_size=16)
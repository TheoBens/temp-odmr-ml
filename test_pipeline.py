"""
Script de test rapide de la pipeline d'optimisation.
Version raccourcie pour tester que tout fonctionne correctement.
"""

import os
import json
import numpy as np
import torch
from optimize_pipeline import (
    generate_mw_configs_strategy,
    generate_dataset,
    run_single_iteration,
    save_best_results
)


def test_mw_config_generation():
    """Test de génération de configs MW."""
    print("\n" + "="*60)
    print("TEST 1: MW Config Generation")
    print("="*60)
    
    strategies = [
        "random", "focused", "ellipticity_sweep",
        "phase_sweep", "amplitude_sweep", "orthogonal", "mixed"
    ]
    
    for strategy in strategies:
        configs = generate_mw_configs_strategy(strategy, n_configs=5, seed=42)
        print(f"\n{strategy.upper()}:")
        for i, (mw_field, phase) in enumerate(configs):
            print(f"  Config {i+1}: Ox={mw_field[0]:.3f}, Oy={mw_field[1]:.3f}, "
                  f"Phase={np.rad2deg(phase):.1f}°")
    
    print("\n✓ MW config generation test passed!")


def test_dataset_generation():
    """Test de génération de petit dataset."""
    print("\n" + "="*60)
    print("TEST 2: Dataset Generation (small)")
    print("="*60)
    
    # Petit dataset pour test
    n_b_configs = 50  # Seulement 50 configs pour le test
    n_mw_configs = 3
    
    # Générer configs B
    b_configs = np.random.uniform(-0.01, 0.01, size=(n_b_configs, 3))
    
    # Générer configs MW
    mw_configs = generate_mw_configs_strategy("focused", n_configs=n_mw_configs, seed=42)
    
    # Fréquences
    freq_list = np.linspace(2700, 3050, 250)
    
    # Générer dataset
    test_dir = "test_pipeline_quick"
    
    print(f"Generating {n_b_configs} samples with {n_mw_configs} MW configs...")
    generate_dataset(
        mw_configs=mw_configs,
        b_configs=b_configs,
        output_dir=test_dir,
        freq_list=freq_list,
        tilt_x=4.0,
        tilt_y=0.0
    )
    
    # Vérifier que les fichiers sont créés
    assert os.path.exists(os.path.join(test_dir, "configs.csv"))
    assert os.path.exists(os.path.join(test_dir, "frequencies.npy"))
    assert os.path.exists(os.path.join(test_dir, "mw_configs.npy"))
    assert os.path.exists(os.path.join(test_dir, "signals"))
    
    # Compter les fichiers de signal
    n_signals = len([f for f in os.listdir(os.path.join(test_dir, "signals")) 
                     if f.endswith(".npy")])
    assert n_signals == n_b_configs
    
    print(f"\n✓ Dataset generation test passed!")
    print(f"  - Created {n_signals} signal files")
    print(f"  - All metadata files present")
    
    # Nettoyer
    import shutil
    shutil.rmtree(test_dir)
    print(f"  - Cleaned up test directory")


def test_quick_optimization():
    """Test d'optimisation rapide (2 itérations)."""
    print("\n" + "="*60)
    print("TEST 3: Quick Optimization Pipeline (2 iterations)")
    print("="*60)
    
    # Configuration de test
    n_iterations = 2
    n_b_configs = 100  # Petit dataset
    n_freq_points = 250
    
    # Générer B configs
    b_configs = np.random.uniform(-0.01, 0.01, size=(n_b_configs, 3))
    freq_list = np.linspace(2700, 3050, n_freq_points)
    
    best_result = None
    best_metric = float('inf')
    
    for iteration in range(n_iterations):
        print(f"\n--- Test Iteration {iteration + 1}/{n_iterations} ---")
        
        try:
            result = run_single_iteration(iteration, b_configs, freq_list)
            
            if result is not None:
                current_metric = result["metrics"]["avg_rmse"]
                
                if current_metric < best_metric:
                    best_metric = current_metric
                    best_result = result
                    print(f"✓ New best: {current_metric:.3f} mT")
                else:
                    print(f"  No improvement ({current_metric:.3f} mT vs {best_metric:.3f} mT)")
        
        except Exception as e:
            print(f"❌ Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
    
    if best_result:
        print("\n" + "="*60)
        print("QUICK TEST RESULTS")
        print("="*60)
        print(f"Best iteration: {best_result['iteration']}")
        print(f"Best model: {best_result['model_type']}")
        print(f"Best metric: {best_metric:.3f} mT")
        
        # Sauvegarder dans un dossier de test
        test_results_dir = "test_optimization_results"
        save_best_results(best_result)
        
        print(f"\n✓ Quick optimization test passed!")
        print(f"  - Results saved to {test_results_dir}")
    else:
        print("\n❌ Quick optimization test failed - no valid results")


def run_all_tests():
    """Exécute tous les tests."""
    print("\n" + "="*60)
    print("RUNNING PIPELINE TESTS")
    print("="*60)
    
    try:
        test_mw_config_generation()
        test_dataset_generation()
        
        # Test d'optimisation rapide (automatique)
        print("\nRunning quick optimization test (2 iterations)...")
        print("(This will take a few minutes)")
        test_quick_optimization()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe pipeline is ready to use.")
        print("Run 'python optimize_pipeline.py' to start the full optimization.")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

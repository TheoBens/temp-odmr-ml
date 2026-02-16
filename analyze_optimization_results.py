"""
Script pour analyser et visualiser les résultats de l'optimisation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load_optimization_results(results_dir="best_optimization_results"):
    """Charge les résultats de l'optimisation."""
    
    results_file = os.path.join(results_dir, "best_results.json")
    history_file = os.path.join(results_dir, "optimization_history.json")
    
    if not os.path.exists(results_file):
        print(f"No results found in {results_dir}")
        return None, None
    
    with open(results_file, "r") as f:
        best_results = json.load(f)
    
    history = None
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    
    return best_results, history


def print_best_results(best_results):
    """Affiche les meilleurs résultats."""
    print("\n" + "="*70)
    print("BEST OPTIMIZATION RESULTS")
    print("="*70)
    
    print(f"\nIteration: {best_results['iteration']}")
    print(f"Strategy: {best_results['strategy']}")
    print(f"Model: {best_results['model_type']}")
    print(f"Parameters: {best_results['n_parameters']:,}")
    print(f"Timestamp: {best_results['timestamp']}")
    
    print(f"\n{'PERFORMANCE METRICS':^70}")
    print("-"*70)
    metrics = best_results['metrics']
    print(f"RMSE |B|           : {metrics['rmse_norm']:.3f} mT")
    print(f"RMSE Bx            : {metrics['rmse_bx']:.3f} mT")
    print(f"RMSE By            : {metrics['rmse_by']:.3f} mT")
    print(f"RMSE Bz            : {metrics['rmse_bz']:.3f} mT")
    print(f"Average RMSE       : {metrics['avg_rmse']:.3f} mT")
    print(f"Mean angular error : {metrics['mean_angle_deg']:.2f}°")
    
    print(f"\n{'OPTIMAL MW CONFIGURATIONS':^70}")
    print("-"*70)
    for i, config in enumerate(best_results['mw_configs']):
        print(f"Config {i+1}: Ox={config['ox']:.3f}, Oy={config['oy']:.3f}, "
              f"Phase={config['phase_deg']:.1f}°")
    
    print("="*70)


def plot_optimization_history(history, save_dir="best_optimization_results"):
    """Trace l'évolution des métriques au cours de l'optimisation."""
    
    if history is None or len(history) == 0:
        print("No history to plot")
        return
    
    iterations = [h['iteration'] for h in history]
    metrics = [h['metric'] for h in history]
    strategies = [h['strategy'] for h in history]
    models = [h['model_type'] for h in history]
    
    # Trouver le meilleur à chaque itération (minimum cumulatif)
    best_so_far = []
    current_best = float('inf')
    for m in metrics:
        if m < current_best:
            current_best = m
        best_so_far.append(current_best)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Métrique au fil des itérations
    ax1 = axes[0]
    ax1.plot(iterations, metrics, 'o-', label='Current iteration', alpha=0.6)
    ax1.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best so far')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Average RMSE (mT)')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution par stratégie
    ax2 = axes[1]
    
    # Grouper par stratégie
    strategy_metrics = {}
    for s, m in zip(strategies, metrics):
        if s not in strategy_metrics:
            strategy_metrics[s] = []
        strategy_metrics[s].append(m)
    
    strategy_names = list(strategy_metrics.keys())
    strategy_means = [np.mean(strategy_metrics[s]) for s in strategy_names]
    strategy_stds = [np.std(strategy_metrics[s]) if len(strategy_metrics[s]) > 1 else 0 
                     for s in strategy_names]
    
    x_pos = np.arange(len(strategy_names))
    ax2.bar(x_pos, strategy_means, yerr=strategy_stds, alpha=0.7, capsize=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax2.set_ylabel('Average RMSE (mT)')
    ax2.set_title('Performance by Strategy')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "optimization_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nHistory plot saved to {save_path}")
    plt.show()


def plot_mw_configs_distribution(best_results, save_dir="best_optimization_results"):
    """Visualise les configurations MW optimales."""
    
    mw_configs = best_results['mw_configs']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Amplitudes Ox vs Oy
    ax1 = axes[0]
    ox_values = [c['ox'] for c in mw_configs]
    oy_values = [c['oy'] for c in mw_configs]
    
    ax1.scatter(ox_values, oy_values, s=200, alpha=0.6, c=range(len(mw_configs)), 
                cmap='viridis', edgecolors='black', linewidth=2)
    
    for i, (ox, oy) in enumerate(zip(ox_values, oy_values)):
        ax1.annotate(f'{i+1}', (ox, oy), ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='white')
    
    ax1.set_xlabel('Ox (amplitude X)', fontsize=12)
    ax1.set_ylabel('Oy (amplitude Y)', fontsize=12)
    ax1.set_title('MW Amplitudes Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Distribution des phases
    ax2 = axes[1]
    phases = [c['phase_deg'] for c in mw_configs]
    
    # Polar plot
    ax2 = plt.subplot(122, projection='polar')
    theta = np.deg2rad(phases)
    radii = np.ones(len(phases))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(mw_configs)))
    ax2.scatter(theta, radii, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2)
    
    for i, (t, r) in enumerate(zip(theta, radii)):
        ax2.annotate(f'{i+1}', (t, r), ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')
    
    ax2.set_ylim(0, 1.2)
    ax2.set_title('MW Phase Distribution', fontsize=14, fontweight='bold', pad=20)
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "optimal_mw_configs.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"MW configs plot saved to {save_path}")
    plt.show()


def plot_metrics_comparison(best_results, save_dir="best_optimization_results"):
    """Compare les différentes métriques."""
    
    metrics = best_results['metrics']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: RMSE par composante
    ax1 = axes[0]
    components = ['Bx', 'By', 'Bz', '|B|', 'Avg']
    rmse_values = [
        metrics['rmse_bx'],
        metrics['rmse_by'],
        metrics['rmse_bz'],
        metrics['rmse_norm'],
        metrics['avg_rmse']
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax1.bar(components, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax1.set_ylabel('RMSE (mT)', fontsize=12)
    ax1.set_title('RMSE by Component', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Ajouter valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Erreur angulaire
    ax2 = axes[1]
    angle_error = metrics['mean_angle_deg']
    
    # Gauge-style visualization
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    ax2 = plt.subplot(122, projection='polar')
    ax2.plot(theta, r, 'k-', linewidth=2)
    ax2.fill_between(theta, 0, r, alpha=0.1, color='gray')
    
    # Ajouter l'aiguille pour l'erreur angulaire
    max_angle = 90  # degrés
    angle_norm = min(angle_error / max_angle, 1.0)
    angle_rad = angle_norm * np.pi
    
    ax2.plot([angle_rad, angle_rad], [0, 1], 'r-', linewidth=4)
    ax2.scatter([angle_rad], [1], s=200, c='red', zorder=5, edgecolors='black', linewidth=2)
    
    ax2.set_ylim(0, 1.2)
    ax2.set_theta_zero_location('W')
    ax2.set_theta_direction(1)
    ax2.set_xticks(np.linspace(0, np.pi, 7))
    ax2.set_xticklabels(['0°', '15°', '30°', '45°', '60°', '75°', '90°'])
    ax2.set_yticks([])
    ax2.set_title(f'Angular Error: {angle_error:.2f}°', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "metrics_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to {save_path}")
    plt.show()


def generate_optimization_report(results_dir="best_optimization_results"):
    """Génère un rapport complet d'optimisation."""
    
    best_results, history = load_optimization_results(results_dir)
    
    if best_results is None:
        print("No results to analyze")
        return
    
    # Afficher les résultats
    print_best_results(best_results)
    
    # Générer les graphiques
    if history:
        plot_optimization_history(history, results_dir)
    
    plot_mw_configs_distribution(best_results, results_dir)
    plot_metrics_comparison(best_results, results_dir)
    
    # Statistiques supplémentaires
    if history:
        print("\n" + "="*70)
        print("OPTIMIZATION STATISTICS")
        print("="*70)
        
        metrics = [h['metric'] for h in history]
        print(f"Total iterations: {len(history)}")
        print(f"Best metric: {min(metrics):.3f} mT")
        print(f"Worst metric: {max(metrics):.3f} mT")
        print(f"Mean metric: {np.mean(metrics):.3f} mT")
        print(f"Std metric: {np.std(metrics):.3f} mT")
        print(f"Improvement: {max(metrics) - min(metrics):.3f} mT ({(max(metrics) - min(metrics))/max(metrics)*100:.1f}%)")
        
        # Stratégies utilisées
        strategies = list(set([h['strategy'] for h in history]))
        print(f"\nStrategies tested: {len(strategies)}")
        for s in strategies:
            count = sum(1 for h in history if h['strategy'] == s)
            avg_metric = np.mean([h['metric'] for h in history if h['strategy'] == s])
            print(f"  - {s}: {count} iterations, avg={avg_metric:.3f} mT")
        
        # Modèles testés
        models = list(set([h['model_type'] for h in history]))
        print(f"\nModels tested: {len(models)}")
        for m in models:
            count = sum(1 for h in history if h['model_type'] == m)
            avg_metric = np.mean([h['metric'] for h in history if h['model_type'] == m])
            print(f"  - {m}: {count} iterations, avg={avg_metric:.3f} mT")
        
        print("="*70)


if __name__ == "__main__":
    generate_optimization_report()

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

def plot_spearman_correlation(
    records: List[Dict[str, Any]],
    score_key: str,
    detector_name: str,
    rho: float,
    save_path: str = "./results/spearman_correlation.png"
):
    """
    Plots Shift Score vs Ground-truth Performance Drop with Spearman rho.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x = [r[score_key] for r in records]
    y = [r["true_drop"] for r in records]
    domains = [r["domain"] for r in records]

    plt.figure(figsize=(8, 6))
    unique_domains = list(set(domains))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
    domain_to_color = dict(zip(unique_domains, colors))

    for d in unique_domains:
        d_x = [x[i] for i, dom in enumerate(domains) if dom == d]
        d_y = [y[i] for i, dom in enumerate(domains) if dom == d]
        plt.scatter(d_x, d_y, label=d, color=domain_to_color[d], alpha=0.8, s=60)

    # Plot trend line if variance exists
    if len(x) > 1 and np.std(x) > 1e-6:
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(min(x), max(x), 100)
        plt.plot(x_line, m * x_line + b, color='black', linestyle='--', label=f'Linear Fit (Slope={m:.2f})')

    plt.xlabel(f"Unsupervised Shift Score ({detector_name})", fontsize=11)
    plt.ylabel("Ground-Truth Performance Drop (Acc_src - Acc_tgt)", fontsize=11)
    plt.title(f"Radar Signal vs True Drop | Spearman $\\rho = {rho:.4f}$", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved correlation plot to: {save_path}")

def plot_shift_severity_curve(
    records: List[Dict[str, Any]],
    save_path: str = "./results/severity_curves.png"
):
    """
    Plots true accuracy drop across shift severity levels for each domain.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    domain_groups = {}
    for r in records:
        dom = r["domain"]
        if dom not in domain_groups:
            domain_groups[dom] = []
        domain_groups[dom].append((r["severity"], r["true_drop"]))

    plt.figure(figsize=(9, 6))
    for dom, pairs in domain_groups.items():
        pairs.sort(key=lambda p: p[0])
        sevs = [p[0] for p in pairs]
        drops = [p[1] for p in pairs]
        plt.plot(sevs, drops, marker='o', label=dom)

    plt.xlabel("Corruption Severity Level (1 - 5)", fontsize=11)
    plt.ylabel("Performance Drop", fontsize=11)
    plt.title("Performance Drop across Shift Levels", fontsize=13, fontweight='bold')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved severity curve to: {save_path}")

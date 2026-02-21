"""
Result Visualisation  (Section VII figures)
===========================================

Generates publication-quality figures:
  1. Predictions with uncertainty bands
  2. Reliability (calibration) diagram
  3. ROC curve
  4. Uncertainty distribution by class
  5. Kernel variance attribution
  6. Temporal-scale performance
  7. Adversarial robustness curve
  8. Training curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


class ResultVisualizer:
    """Generate all figures for the paper / notebook.

    Parameters
    ----------
    output_dir : str
        Directory to save figures.
    """

    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_all(
        self,
        results: Dict,
        train_metrics: Dict,
        X_sample=None,
        y_sample=None,
        detector=None,
        domain: str = "multi",
    ):
        """Generate the full 3x3 figure grid."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # Row 0
        self.plot_predictions_with_uncertainty(
            axes[0, 0], detector, X_sample, y_sample, domain
        )
        self.plot_training_curves(axes[0, 1], train_metrics)
        self.plot_roc_curve(axes[0, 2], results)

        # Row 1
        self.plot_reliability_diagram(axes[1, 0], results.get("calibration", {}))
        self.plot_uncertainty_distribution(
            axes[1, 1], detector, X_sample, y_sample, domain
        )
        self.plot_kernel_attribution(axes[1, 2], results.get("kernel_attribution", {}))

        # Row 2
        self.plot_temporal_performance(axes[2, 0], results.get("temporal", {}))
        self.plot_adversarial_robustness(axes[2, 1], results.get("adversarial", {}))
        self.plot_alert_distribution(axes[2, 2], results.get("alerts", {}))

        plt.suptitle(
            "Hierarchical GP — Uncertainty-Calibrated Detection Results",
            fontsize=15, y=1.01,
        )
        plt.tight_layout()
        path = os.path.join(self.output_dir, "results_overview.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Saved → {path}")

    # ------------------------------------------------------------------ #
    #  Individual plots                                                    #
    # ------------------------------------------------------------------ #
    def plot_predictions_with_uncertainty(self, ax, detector, X, y, domain):
        if detector is None or X is None:
            ax.set_visible(False)
            return
        import torch
        X_t = torch.as_tensor(X[:200], dtype=torch.float32) if not isinstance(X, torch.Tensor) else X[:200]
        y_np = y[:200].cpu().numpy() if isinstance(y, torch.Tensor) else y[:200]

        res = detector.detect(X_t.to(next(detector.model.parameters()).device), domain)
        mean = res["uncertainties"]["mean"].cpu().numpy()
        std = res["uncertainties"]["std"].cpu().numpy()
        idx = np.arange(len(mean))

        ax.plot(idx, mean, "b-", alpha=0.7, label="Pred mean")
        ax.fill_between(idx, mean - 2 * std, mean + 2 * std, alpha=0.25, color="blue", label="95% CI")
        anom = y_np == 1
        if anom.any():
            ax.scatter(idx[anom], mean[anom], c="red", marker="x", s=40, label="True attack", zorder=5)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Score")
        ax.set_title("Predictions + Uncertainty")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_training_curves(self, ax, train_metrics):
        if not train_metrics.get("train"):
            ax.set_visible(False)
            return
        losses = [m["loss"] for m in train_metrics["train"]]
        ax.plot(losses, "b-", label="Train loss")
        if train_metrics.get("val"):
            ve = [m["epoch"] for m in train_metrics["val"]]
            vl = [m["loss"] for m in train_metrics["val"]]
            ax.plot(ve, vl, "r--o", markersize=3, label="Val loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Neg. ELBO")
        ax.set_title("Training Curves")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_roc_curve(self, ax, results):
        if "roc" not in results:
            ax.set_visible(False)
            return
        fpr, tpr = results["roc"]["fpr"], results["roc"]["tpr"]
        auc_val = results["roc"]["auc"]
        ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_reliability_diagram(self, ax, cal):
        if not cal.get("bins"):
            ax.set_visible(False)
            return
        confs = [b["confidence"] for b in cal["bins"]]
        accs = [b["accuracy"] for b in cal["bins"]]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        ax.bar(confs, accs, width=0.06, alpha=0.6, color="steelblue", label="Model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Calibration  (ECE = {cal.get('ece', 0):.3f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_uncertainty_distribution(self, ax, detector, X, y, domain):
        if detector is None or X is None:
            ax.set_visible(False)
            return
        import torch
        X_t = torch.as_tensor(X[:2000], dtype=torch.float32) if not isinstance(X, torch.Tensor) else X[:2000]
        y_np = y[:2000].cpu().numpy() if isinstance(y, torch.Tensor) else y[:2000]
        dev = next(detector.model.parameters()).device
        unc = detector.compute_uncertainty(X_t.to(dev))
        epist = unc["epistemic"].cpu().numpy()
        normal = epist[y_np == 0]
        attack = epist[y_np == 1]
        if len(normal):
            ax.hist(normal, bins=40, alpha=0.5, density=True, color="blue", label="Normal")
        if len(attack):
            ax.hist(attack, bins=40, alpha=0.5, density=True, color="red", label="Attack")
        ax.set_xlabel("Epistemic Uncertainty")
        ax.set_ylabel("Density")
        ax.set_title("Uncertainty by Class")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def plot_kernel_attribution(self, ax, attr):
        if not attr:
            ax.set_visible(False)
            return
        names = list(attr.keys())
        vals = list(attr.values())
        ax.barh(names, vals, color="steelblue")
        ax.set_xlabel("Variance Fraction")
        ax.set_title("Kernel Variance Attribution")
        ax.grid(True, alpha=0.3, axis="x")

    def plot_temporal_performance(self, ax, temporal):
        if not temporal:
            ax.set_visible(False)
            return
        scales = list(temporal.keys())
        accs = list(temporal.values())
        ax.plot(scales, accs, "bo-", lw=2, ms=7)
        ax.set_ylabel("Accuracy")
        ax.set_title("Temporal-Scale Performance")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    def plot_adversarial_robustness(self, ax, adv):
        if not adv:
            ax.set_visible(False)
            return
        eps = sorted(adv.keys())
        accs = [adv[e] for e in eps]
        ax.plot(eps, accs, "ro-", lw=2, ms=7)
        ax.set_xlabel("Perturbation (eps)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Adversarial Robustness")
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    def plot_alert_distribution(self, ax, alerts):
        if not alerts:
            ax.set_visible(False)
            return
        tiers = list(alerts.keys())
        counts = list(alerts.values())
        colours = {"low": "green", "medium": "orange", "critical": "red"}
        ax.bar(tiers, counts, color=[colours.get(t, "gray") for t in tiers])
        ax.set_ylabel("Count")
        ax.set_title("Alert Tier Distribution")
        ax.grid(True, alpha=0.3, axis="y")

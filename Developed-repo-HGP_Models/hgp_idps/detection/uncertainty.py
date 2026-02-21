"""
Uncertainty-Calibrated Detection  (Section IV.D)
================================================

Anomaly scoring with epistemic / aleatoric decomposition, domain-adaptive
thresholds, and imbalance compensation per Eq. 39-40.
"""

import torch
import gpytorch
import numpy as np
from typing import Dict, Optional


class UncertaintyCalibratedDetector:
    """Uncertainty-aware anomaly scoring and adaptive thresholding.

    Parameters
    ----------
    model : DeepKernelHGP
        Trained deep-kernel hierarchical GP.
    config : dict
        Detection hyper-parameters (see configs/default_config.yaml).
    """

    IMBALANCE_RATIOS = {
        "edge_iiot": 2.67,
        "container": 15.7,
        "soc": 99.0,
        "multi": 10.0,
    }

    def __init__(self, model, config: Optional[dict] = None):
        self.model = model
        self.cfg = config or {
            "uncertainty_weight": 0.5,
            "entropy_weight": 0.3,
            "adaptive_threshold": True,
            "base_threshold": 0.5,
        }
        self.baseline_stats: Optional[Dict] = None

    # ------------------------------------------------------------------ #
    #  Core uncertainty decomposition  (Section IV.D.1)                    #
    # ------------------------------------------------------------------ #
    def compute_uncertainty(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decompose predictive uncertainty into epistemic + aleatoric.

        H[Y|x] = E_θ[H[Y|x,θ]]  +  I[Y;θ|x]
                  (aleatoric)         (epistemic)
        """
        self.model.eval()
        self.model.gp.eval()
        self.model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Latent GP posterior
            f_dist = self.model(X)
            f_mean = f_dist.mean
            f_var = f_dist.variance          # epistemic (model) variance

            # Predictive distribution through likelihood
            pred = self.model.likelihood(f_dist)
            p = pred.mean.clamp(1e-6, 1 - 1e-6)
            pred_var = pred.variance

            # Aleatoric = total − epistemic
            aleatoric = (pred_var - f_var).clamp(min=0)

            # Predictive entropy  H[Y|x]
            entropy = -p * p.log() - (1 - p) * (1 - p).log()

            # Confidence
            confidence = 1.0 / (1.0 + pred_var.sqrt())

        return {
            "mean": p,
            "f_mean": f_mean,
            "variance": pred_var,
            "std": pred_var.sqrt(),
            "epistemic": f_var,
            "aleatoric": aleatoric,
            "entropy": entropy,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------ #
    #  Anomaly score  (Eq. 39)                                             #
    # ------------------------------------------------------------------ #
    def anomaly_score(
        self, X: torch.Tensor, domain: str = "multi"
    ) -> Dict[str, torch.Tensor]:
        """Domain-adaptive anomaly scoring.

        s^(d)(x,t) = |μ − μ_baseline| / √(σ² + σ²_noise / ρ_d) + λ·H(x,t)
        """
        unc = self.compute_uncertainty(X)
        rho = self.IMBALANCE_RATIOS.get(domain, 10.0)

        deviation = (
            torch.abs(unc["mean"] - self.baseline_stats["mean"])
            if self.baseline_stats
            else unc["mean"]
        )

        noise_var = 0.1
        denom = (unc["variance"] + noise_var / rho).sqrt().clamp(min=1e-6)
        norm_score = deviation / denom

        lam = self.cfg["entropy_weight"]
        score = norm_score + lam * unc["entropy"]

        return {"score": score, "uncertainties": unc, "deviation": deviation}

    # ------------------------------------------------------------------ #
    #  Adaptive threshold  (Eq. 40)                                        #
    # ------------------------------------------------------------------ #
    def adaptive_threshold(
        self, unc: Dict[str, torch.Tensor], domain: str = "multi"
    ) -> torch.Tensor:
        """τ^(d)(x,t) = τ_0 + γ·σ·√ρ_d + β·H(x,t)."""
        rho = self.IMBALANCE_RATIOS.get(domain, 10.0)
        tau0 = self.cfg["base_threshold"]
        gamma, beta = 0.3, 0.2
        return tau0 + gamma * unc["std"] * np.sqrt(rho) + beta * unc["entropy"]

    # ------------------------------------------------------------------ #
    #  Full detection pipeline                                             #
    # ------------------------------------------------------------------ #
    def detect(
        self, X: torch.Tensor, domain: str = "multi"
    ) -> Dict[str, torch.Tensor]:
        res = self.anomaly_score(X, domain)
        unc = res["uncertainties"]

        if self.cfg.get("adaptive_threshold", True):
            thresh = self.adaptive_threshold(unc, domain)
        else:
            thresh = torch.full_like(res["score"], self.cfg["base_threshold"])

        detections = (res["score"] > thresh).float()
        return {
            "detections": detections,
            "scores": res["score"],
            "threshold": thresh,
            "uncertainties": unc,
            "confidence": unc["confidence"],
        }

    # ------------------------------------------------------------------ #
    #  Baseline calibration                                                #
    # ------------------------------------------------------------------ #
    def update_baseline(self, X: torch.Tensor):
        """Compute baseline statistics from known-normal data."""
        unc = self.compute_uncertainty(X)
        self.baseline_stats = {
            "mean": unc["mean"].mean(),
            "std": unc["std"].mean(),
            "variance": unc["variance"].mean(),
        }

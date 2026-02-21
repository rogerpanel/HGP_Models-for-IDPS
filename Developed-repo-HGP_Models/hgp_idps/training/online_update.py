"""
Incremental Posterior Updates  (Appendix G.1)
=============================================

Streaming data incorporation via recursive variational update:

  m_{t+1} = m_t + η_d · K_{*u} K_{uu}^{-1} (y_t - μ_t)
  S_{t+1} = S_t + η_d · (K_{*u} K_{uu}^{-1} K_{u*} - S_t)

Learning rate adapts via  η_d = η_0 / √ρ_d  to account for
domain-specific imbalance ratios.
"""

import math
import torch
import gpytorch
from typing import Optional


class IncrementalPosteriorUpdater:
    """Online update of variational parameters without full retraining.

    Parameters
    ----------
    model : DeepKernelHGP
        Trained model whose variational parameters we update.
    base_lr : float
        Base learning rate η_0 (scaled by √ρ_d).
    """

    IMBALANCE = {"edge_iiot": 2.67, "container": 15.7, "soc": 99.0, "multi": 10.0}

    def __init__(self, model, base_lr: float = 0.01):
        self.model = model
        self.base_lr = base_lr

    def update(
        self,
        x_new: torch.Tensor,
        y_new: torch.Tensor,
        domain: str = "multi",
    ):
        """Incorporate a small batch of new observations.

        Parameters
        ----------
        x_new : Tensor [B, D]
            New raw-feature observations.
        y_new : Tensor [B]
            Corresponding labels.
        domain : str
            Domain identifier for adaptive learning rate.
        """
        rho = self.IMBALANCE.get(domain, 10.0)
        eta = self.base_lr / math.sqrt(rho)

        gp = self.model.gp
        gp.eval()
        self.model.projection.eval()
        self.model.likelihood.eval()

        with torch.no_grad():
            z_new = self.model.projection(x_new)

            # Current variational parameters
            var_dist = gp.variational_strategy.variational_distribution
            m = var_dist.mean.clone()           # [M]
            S = var_dist.covariance_matrix.clone()  # [M, M]

            # Inducing locations
            Z = gp.variational_strategy.inducing_points  # [M, d]

            # Kernel evaluations
            K_uu = gp.covar_module(Z, Z).to_dense()        # [M, M]
            K_uu_inv = torch.linalg.solve(
                K_uu + 1e-4 * torch.eye(K_uu.size(0), device=K_uu.device),
                torch.eye(K_uu.size(0), device=K_uu.device),
            )

            K_star_u = gp.covar_module(z_new, Z).to_dense()  # [B, M]

            # Current predictive mean at new points
            mu_new = (K_star_u @ K_uu_inv @ m)  # [B]

            # Residuals
            residual = y_new.float() - mu_new  # [B]

            # Update mean:  m ← m + η · K_{uu}^{-1} K_{u*} residual
            update_m = K_uu_inv @ K_star_u.t() @ residual  # [M]
            m_new = m + eta * update_m

            # Update covariance:  S ← S + η · (K_{*u}^T K_{uu}^{-1} K_{u*} - S)
            correction = K_star_u.t() @ K_star_u  # [M, M]
            correction = K_uu_inv @ correction @ K_uu_inv
            S_new = S + eta * (correction - S)

            # Ensure positive-definiteness
            S_new = 0.5 * (S_new + S_new.t())
            eigvals = torch.linalg.eigvalsh(S_new)
            if eigvals.min() < 1e-6:
                S_new += (1e-6 - eigvals.min()) * torch.eye(
                    S_new.size(0), device=S_new.device
                )

            # Write back
            var_dist.variational_mean.data.copy_(m_new)
            # Update Cholesky factor
            L_new = torch.linalg.cholesky(S_new)
            var_dist.chol_variational_covar.data.copy_(L_new)

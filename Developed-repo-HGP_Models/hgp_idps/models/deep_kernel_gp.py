"""
Deep-Kernel Hierarchical GP  (combines projection + GP)
========================================================

End-to-end model that chains:
  1.  ProjectionNetwork  π(·)   – feature extraction
  2.  HierarchicalCloudSecurityGP  – variational GP in the latent space

This is the top-level model used for training and inference.
"""

import torch
import torch.nn as nn
import gpytorch
from gpytorch.likelihoods import BernoulliLikelihood

from .projection_network import ProjectionNetwork
from .hierarchical_gp import HierarchicalCloudSecurityGP


class DeepKernelHGP(nn.Module):
    """End-to-end Deep-Kernel Hierarchical GP.

    Parameters
    ----------
    projection : ProjectionNetwork
        Pre-built projection network  π.
    gp : HierarchicalCloudSecurityGP
        Pre-built sparse variational GP operating in π's output space.
    likelihood : BernoulliLikelihood
        Bernoulli likelihood for binary classification.
    """

    def __init__(
        self,
        projection: ProjectionNetwork,
        gp: HierarchicalCloudSecurityGP,
        likelihood: BernoulliLikelihood,
    ):
        super().__init__()
        self.projection = projection
        self.gp = gp
        self.likelihood = likelihood

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Project features then compute GP posterior."""
        z = self.projection(x)
        return self.gp(z)

    # ------------------------------------------------------------------
    def predict(self, x: torch.Tensor):
        """Return likelihood-transformed predictive distribution."""
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_dist = self.forward(x)
            return self.likelihood(f_dist)

    # ------------------------------------------------------------------
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent features (useful for inducing-point init)."""
        self.projection.eval()
        with torch.no_grad():
            return self.projection(x)

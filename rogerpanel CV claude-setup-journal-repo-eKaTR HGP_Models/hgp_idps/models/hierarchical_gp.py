"""
Hierarchical GP  (Section IV.A-B, Proposition 1)
=================================================

Sparse variational GP with hierarchical additive kernel decomposition:

  f(x^(d), t) = f_shared(π(x^(d)), t) + f_domain^(d)(x^(d), t)
              + f_interact^(d)(x^(d), t)  + r_K(x^(d), t)

Uses Cholesky variational distribution with unwhitened variational strategy
for O(N_b M^2 + M^3) per-epoch training and O(M) prediction.
"""

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
)
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.distributions import MultivariateNormal


class HierarchicalCloudSecurityGP(ApproximateGP):
    """Sparse variational GP with hierarchical additive kernel.

    Parameters
    ----------
    inducing_points : Tensor  [M, d_z]
        Initial locations of the M inducing inputs in the projected space.
    feature_dim : int
        Dimensionality d_z of the projected features (output of π).
    kernel : gpytorch.kernels.Kernel
        Pre-built additive kernel (shared + domain + temporal + interaction).
    use_linear_mean : bool
        If True use LinearMean (for cross-domain), else ConstantMean.
    learn_inducing : bool
        Whether inducing locations are optimised during training.
    """

    def __init__(
        self,
        inducing_points: torch.Tensor,
        feature_dim: int,
        kernel: gpytorch.kernels.Kernel,
        use_linear_mean: bool = True,
        learn_inducing: bool = True,
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing,
        )
        super().__init__(variational_strategy)

        self.feature_dim = feature_dim
        self.mean_module = (
            LinearMean(feature_dim) if use_linear_mean else ConstantMean()
        )
        self.covar_module = kernel

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
